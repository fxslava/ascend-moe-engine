/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// safetensors_view.hpp -- a zero-dependency, header-only, read-only
// safetensors reader for the host emulation layer.
//
// THE FORMAT
//   [0, 8)              uint64 header_len, little-endian
//   [8, 8 + header_len) UTF-8 JSON header
//   [8 + header_len, N) the tensor data segment
//
// Every entry maps a tensor name to {dtype, shape, data_offsets:[start,end]},
// where the offsets are relative to the START OF THE DATA SEGMENT, not to the
// start of the file. The optional `__metadata__` key maps to a string->string
// object and is NOT a tensor; it is kept separately.
//
// TRUNCATED FILES ARE FIRST-CLASS
// -------------------------------
// `scripts/download_dsv4_layer0.sh` fetches only a leading window of a 3.5 GB
// shard, because every tensor the Layer 0 emulation reads lives in the first
// ~11.5 MB of it. The resulting file has a COMPLETE header describing 1565
// tensors and the bytes for perhaps 30 of them. That is a legitimate input
// here, not a corrupt file: the header still parses in full, and each
// `TensorView` carries `available` telling you whether its bytes are actually
// mapped. `Tensor(name)` throws for a tensor outside the window -- naming the
// byte range it wanted and the file size it got -- while `Find(name)` hands
// back the unavailable view so a caller can report the gap instead.
//
// Reading past EOF through a shorter-than-header mapping would be a SIGBUS,
// not a short read, so this distinction is what keeps the bench from dying on
// a sliced shard.
//
// ZERO COPY
// ---------
// `data_ptr` points straight into the mmap. Nothing is decoded, byte-swapped
// or converted on load: a BF16 [24, 16384] tensor costs one page-fault walk
// and no heap. The mapping outlives every view, so a `SafetensorsView` must
// outlive the pointers it handed out.
//
// ALIGNMENT
// ---------
// The data segment begins at 8 + header_len, which safetensors does not
// require to be aligned to the element size, so `data_ptr` may be misaligned
// for its dtype. Read multi-byte elements through the `ElementAt<T>()` /
// `CopyTo<T>()` helpers (which memcpy) rather than by casting the pointer --
// a cast would be undefined behaviour and, on the FP32 [24, 16384] mHC phi,
// genuinely unaligned in this checkpoint.

#ifndef MOE_EMU_SAFETENSORS_VIEW_HPP
#define MOE_EMU_SAFETENSORS_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace moe::emu {

// ---------------------------------------------------------------------------
// dtypes
// ---------------------------------------------------------------------------

// The safetensors dtype spellings, plus the three FP8 ones a DeepSeek-V4
// checkpoint uses. Returns 0 for an unrecognized name, which the parser
// reports rather than guessing a width for.
inline std::size_t SafetensorsDtypeSize(const std::string& dtype) noexcept {
  if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" ||
      dtype == "F8_E4M3" || dtype == "F8_E5M2" || dtype == "F8_E8M0") {
    return 1;
  }
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") {
    return 2;
  }
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") {
    return 4;
  }
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") {
    return 8;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// TensorView
// ---------------------------------------------------------------------------

struct TensorView {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  const void* data_ptr = nullptr;  // null when !available
  std::size_t byte_size = 0;       // the header's end - start, always

  // False when the header describes this tensor but its bytes fall past the
  // end of the file -- i.e. a sliced shard. See the header comment.
  bool available = false;

  // Byte offsets relative to the start of the DATA SEGMENT, as the header
  // spells them.
  std::uint64_t data_begin = 0;
  std::uint64_t data_end = 0;

  std::size_t element_size() const noexcept { return SafetensorsDtypeSize(dtype); }

  std::int64_t elements() const noexcept {
    std::int64_t total = 1;
    for (const std::int64_t extent : shape) {
      total *= extent;
    }
    return shape.empty() ? 1 : total;
  }

  bool is_shape(std::initializer_list<std::int64_t> expected) const noexcept {
    return shape.size() == expected.size() &&
           std::equal(shape.begin(), shape.end(), expected.begin());
  }

  std::string shape_string() const {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i != 0) {
        out += ", ";
      }
      out += std::to_string(shape[i]);
    }
    return out + "]";
  }

  // One element, read with memcpy so a misaligned data segment is safe.
  template <typename T>
  T ElementAt(std::size_t index) const {
    RequireAvailable();
    if (sizeof(T) != element_size()) {
      throw std::runtime_error("safetensors: tensor '" + name + "' is " + dtype +
                               " (" + std::to_string(element_size()) +
                               " bytes/element); asked for a " +
                               std::to_string(sizeof(T)) + "-byte element");
    }
    if ((index + 1) * sizeof(T) > byte_size) {
      throw std::out_of_range("safetensors: element " + std::to_string(index) +
                              " is past the end of '" + name + "'");
    }
    T value{};
    std::memcpy(&value, static_cast<const std::uint8_t*>(data_ptr) + index * sizeof(T),
                sizeof(T));
    return value;
  }

  // The whole tensor into caller storage, again via memcpy.
  template <typename T>
  void CopyTo(T* destination, std::size_t count) const {
    RequireAvailable();
    if (sizeof(T) != element_size()) {
      throw std::runtime_error("safetensors: tensor '" + name + "' is " + dtype +
                               "; asked to copy " + std::to_string(sizeof(T)) +
                               "-byte elements");
    }
    if (count * sizeof(T) > byte_size) {
      throw std::out_of_range("safetensors: asked for " + std::to_string(count) +
                              " elements of '" + name + "' but it holds " +
                              std::to_string(byte_size / sizeof(T)));
    }
    std::memcpy(destination, data_ptr, count * sizeof(T));
  }

  void RequireAvailable() const {
    if (!available || data_ptr == nullptr) {
      throw std::runtime_error(
          "safetensors: tensor '" + name + "' (" + dtype + " " + shape_string() +
          ", data bytes [" + std::to_string(data_begin) + ", " +
          std::to_string(data_end) +
          ") of the data segment) is described by the header but its bytes are "
          "not in this file -- it is a truncated slice. Re-fetch with "
          "scripts/download_dsv4_layer0.sh --full.");
    }
  }
};

// ---------------------------------------------------------------------------
// A tiny JSON scanner, shaped for the safetensors header
// ---------------------------------------------------------------------------
//
// Not a general JSON library: it handles exactly the value kinds a
// safetensors header can hold (object, array, string, number, true/false/
// null), with full string-escape handling, and it can skip any value it does
// not care about. Everything is bounds-checked against the header length, so
// a malformed or hostile header produces a thrown std::runtime_error naming
// the byte offset rather than a read past the buffer.

namespace detail {

class JsonScanner {
 public:
  JsonScanner(const char* text, std::size_t size) : text_(text), size_(size) {}

  std::size_t position() const noexcept { return position_; }

  void SkipWhitespace() noexcept {
    while (position_ < size_ && (text_[position_] == ' ' || text_[position_] == '\t' ||
                                 text_[position_] == '\n' || text_[position_] == '\r')) {
      ++position_;
    }
  }

  bool AtEnd() noexcept {
    SkipWhitespace();
    return position_ >= size_;
  }

  char Peek() {
    SkipWhitespace();
    if (position_ >= size_) {
      Fail("unexpected end of header");
    }
    return text_[position_];
  }

  void Expect(char expected) {
    const char actual = Peek();
    if (actual != expected) {
      Fail(std::string("expected '") + expected + "' but found '" + actual + "'");
    }
    ++position_;
  }

  bool TryConsume(char candidate) {
    SkipWhitespace();
    if (position_ < size_ && text_[position_] == candidate) {
      ++position_;
      return true;
    }
    return false;
  }

  std::string ParseString() {
    Expect('"');
    std::string out;
    while (true) {
      if (position_ >= size_) {
        Fail("unterminated string");
      }
      const char c = text_[position_++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (position_ >= size_) {
        Fail("unterminated escape");
      }
      const char escape = text_[position_++];
      switch (escape) {
        case '"':  out.push_back('"');  break;
        case '\\': out.push_back('\\'); break;
        case '/':  out.push_back('/');  break;
        case 'b':  out.push_back('\b'); break;
        case 'f':  out.push_back('\f'); break;
        case 'n':  out.push_back('\n'); break;
        case 'r':  out.push_back('\r'); break;
        case 't':  out.push_back('\t'); break;
        case 'u': {
          // \uXXXX -> UTF-8. Surrogate pairs are combined; a lone surrogate
          // is passed through as the replacement character rather than
          // producing invalid UTF-8.
          std::uint32_t code = ParseHex4();
          if (code >= 0xD800U && code <= 0xDBFFU && position_ + 1 < size_ &&
              text_[position_] == '\\' && text_[position_ + 1] == 'u') {
            position_ += 2;
            const std::uint32_t low = ParseHex4();
            if (low >= 0xDC00U && low <= 0xDFFFU) {
              code = 0x10000U + ((code - 0xD800U) << 10U) + (low - 0xDC00U);
            } else {
              AppendUtf8(out, 0xFFFDU);
              code = low;
            }
          } else if (code >= 0xD800U && code <= 0xDFFFU) {
            code = 0xFFFDU;
          }
          AppendUtf8(out, code);
          break;
        }
        default:
          Fail(std::string("unknown escape '\\") + escape + "'");
      }
    }
  }

  // Integers are what a safetensors header actually carries (shapes and byte
  // offsets), so this parses the full JSON number grammar but returns an
  // int64 and refuses a fractional or out-of-range one.
  std::int64_t ParseInteger() {
    SkipWhitespace();
    const std::size_t start = position_;
    if (position_ < size_ && (text_[position_] == '-' || text_[position_] == '+')) {
      ++position_;
    }
    const std::size_t digits_start = position_;
    while (position_ < size_ && text_[position_] >= '0' && text_[position_] <= '9') {
      ++position_;
    }
    if (position_ == digits_start) {
      Fail("expected a number");
    }
    if (position_ < size_ && (text_[position_] == '.' || text_[position_] == 'e' ||
                              text_[position_] == 'E')) {
      Fail("expected an integer but found a floating-point number");
    }
    const std::string literal(text_ + start, position_ - start);
    try {
      return std::stoll(literal);
    } catch (const std::exception&) {
      Fail("integer '" + literal + "' is out of range");
    }
    return 0;  // unreachable; Fail throws
  }

  // Advances past any single value, however nested.
  void SkipValue() {
    const char c = Peek();
    switch (c) {
      case '"':
        ParseString();
        return;
      case '{':
      case '[': {
        const char close = (c == '{') ? '}' : ']';
        ++position_;
        if (TryConsume(close)) {
          return;
        }
        while (true) {
          if (c == '{') {
            ParseString();
            Expect(':');
          }
          SkipValue();
          if (TryConsume(close)) {
            return;
          }
          Expect(',');
        }
      }
      case 't': Literal("true");  return;
      case 'f': Literal("false"); return;
      case 'n': Literal("null");  return;
      default: {
        // A number, possibly floating point -- skip the whole token.
        SkipWhitespace();
        const std::size_t start = position_;
        while (position_ < size_) {
          const char d = text_[position_];
          const bool numeric = (d >= '0' && d <= '9') || d == '-' || d == '+' ||
                               d == '.' || d == 'e' || d == 'E';
          if (!numeric) {
            break;
          }
          ++position_;
        }
        if (position_ == start) {
          Fail(std::string("unexpected character '") + c + "'");
        }
        return;
      }
    }
  }

  [[noreturn]] void Fail(const std::string& message) const {
    throw std::runtime_error("safetensors: malformed JSON header at byte " +
                             std::to_string(position_) + ": " + message);
  }

 private:
  void Literal(const char* word) {
    const std::size_t length = std::strlen(word);
    SkipWhitespace();
    if (position_ + length > size_ || std::strncmp(text_ + position_, word, length) != 0) {
      Fail(std::string("expected '") + word + "'");
    }
    position_ += length;
  }

  std::uint32_t ParseHex4() {
    if (position_ + 4 > size_) {
      Fail("truncated \\u escape");
    }
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[position_++];
      value <<= 4U;
      if (c >= '0' && c <= '9') {
        value |= static_cast<std::uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<std::uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<std::uint32_t>(c - 'A' + 10);
      } else {
        Fail(std::string("'") + c + "' is not a hex digit");
      }
    }
    return value;
  }

  static void AppendUtf8(std::string& out, std::uint32_t code) {
    if (code < 0x80U) {
      out.push_back(static_cast<char>(code));
    } else if (code < 0x800U) {
      out.push_back(static_cast<char>(0xC0U | (code >> 6U)));
      out.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
    } else if (code < 0x10000U) {
      out.push_back(static_cast<char>(0xE0U | (code >> 12U)));
      out.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3FU)));
      out.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
    } else {
      out.push_back(static_cast<char>(0xF0U | (code >> 18U)));
      out.push_back(static_cast<char>(0x80U | ((code >> 12U) & 0x3FU)));
      out.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3FU)));
      out.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
    }
  }

  const char* text_;
  std::size_t size_;
  std::size_t position_ = 0;
};

}  // namespace detail

// ---------------------------------------------------------------------------
// SafetensorsView
// ---------------------------------------------------------------------------

class SafetensorsView {
 public:
  SafetensorsView() = default;

  explicit SafetensorsView(const std::string& path) { Open(path); }

  ~SafetensorsView() { Close(); }

  SafetensorsView(const SafetensorsView&) = delete;
  SafetensorsView& operator=(const SafetensorsView&) = delete;

  SafetensorsView(SafetensorsView&& other) noexcept { MoveFrom(std::move(other)); }

  SafetensorsView& operator=(SafetensorsView&& other) noexcept {
    if (this != &other) {
      Close();
      MoveFrom(std::move(other));
    }
    return *this;
  }

  void Open(const std::string& path) {
    Close();
    path_ = path;

    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      throw std::runtime_error("safetensors: cannot open '" + path + "': " +
                               std::string(std::strerror(errno)));
    }
    struct stat info{};
    if (::fstat(fd, &info) != 0) {
      const std::string reason = std::strerror(errno);
      ::close(fd);
      throw std::runtime_error("safetensors: cannot stat '" + path + "': " + reason);
    }
    file_size_ = static_cast<std::size_t>(info.st_size);
    if (file_size_ < kPrefixBytes) {
      ::close(fd);
      throw std::runtime_error("safetensors: '" + path + "' is " +
                               std::to_string(file_size_) +
                               " bytes, too short to hold the 8-byte header length");
    }

    void* mapping = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    // The descriptor is not needed once the mapping exists.
    ::close(fd);
    if (mapping == MAP_FAILED) {
      throw std::runtime_error("safetensors: cannot mmap '" + path + "': " +
                               std::string(std::strerror(errno)));
    }
    mapping_ = static_cast<const std::uint8_t*>(mapping);

    try {
      ParseHeader();
    } catch (...) {
      Close();
      throw;
    }
  }

  void Close() noexcept {
    if (mapping_ != nullptr) {
      ::munmap(const_cast<void*>(static_cast<const void*>(mapping_)), file_size_);
      mapping_ = nullptr;
    }
    file_size_ = 0;
    header_length_ = 0;
    data_segment_ = nullptr;
    data_segment_bytes_ = 0;
    tensors_.clear();
    metadata_.clear();
    path_.clear();
  }

  bool is_open() const noexcept { return mapping_ != nullptr; }
  const std::string& path() const noexcept { return path_; }
  std::size_t file_size() const noexcept { return file_size_; }
  std::uint64_t header_length() const noexcept { return header_length_; }

  // Bytes of data segment actually present. Equals the sum of the tensor
  // extents for a complete file and is short for a downloaded slice.
  std::size_t data_segment_bytes() const noexcept { return data_segment_bytes_; }

  // `__metadata__`, or empty when the header had none.
  const std::map<std::string, std::string>& metadata() const noexcept { return metadata_; }

  std::size_t size() const noexcept { return tensors_.size(); }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& entry : tensors_) {
      out.push_back(entry.first);
    }
    return out;
  }

  bool contains(const std::string& name) const { return tensors_.count(name) != 0; }

  // The view, or nullptr when the header never mentioned this name. A view
  // whose `available` is false IS returned: the header knows the tensor, the
  // file just does not carry its bytes.
  const TensorView* Find(const std::string& name) const {
    const auto found = tensors_.find(name);
    return found == tensors_.end() ? nullptr : &found->second;
  }

  // The view, throwing when the name is absent or its bytes are outside a
  // truncated file.
  const TensorView& Tensor(const std::string& name) const {
    const TensorView* view = Find(name);
    if (view == nullptr) {
      throw std::runtime_error("safetensors: '" + path_ + "' has no tensor named '" +
                               name + "' (it holds " + std::to_string(tensors_.size()) +
                               " tensors)");
    }
    view->RequireAvailable();
    return *view;
  }

  // Same, plus a shape assertion -- the form the Layer 0 bench uses, so a
  // checkpoint whose geometry moved is reported by name and shape instead of
  // being read at the wrong stride.
  const TensorView& TensorWithShape(const std::string& name,
                                    std::initializer_list<std::int64_t> expected) const {
    const TensorView& view = Tensor(name);
    if (!view.is_shape(expected)) {
      std::string wanted = "[";
      std::size_t index = 0;
      for (const std::int64_t extent : expected) {
        if (index++ != 0) {
          wanted += ", ";
        }
        wanted += std::to_string(extent);
      }
      wanted += "]";
      throw std::runtime_error("safetensors: '" + name + "' is " + view.dtype + " " +
                               view.shape_string() + " but this build expects " + wanted);
    }
    return view;
  }

  // Every tensor whose name begins with `prefix`, in name order.
  std::vector<const TensorView*> WithPrefix(const std::string& prefix) const {
    std::vector<const TensorView*> out;
    for (auto it = tensors_.lower_bound(prefix); it != tensors_.end(); ++it) {
      if (it->first.compare(0, prefix.size(), prefix) != 0) {
        break;
      }
      out.push_back(&it->second);
    }
    return out;
  }

 private:
  static constexpr std::size_t kPrefixBytes = 8;

  void MoveFrom(SafetensorsView&& other) noexcept {
    mapping_ = other.mapping_;
    file_size_ = other.file_size_;
    header_length_ = other.header_length_;
    data_segment_ = other.data_segment_;
    data_segment_bytes_ = other.data_segment_bytes_;
    tensors_ = std::move(other.tensors_);
    metadata_ = std::move(other.metadata_);
    path_ = std::move(other.path_);
    other.mapping_ = nullptr;
    other.file_size_ = 0;
    other.header_length_ = 0;
    other.data_segment_ = nullptr;
    other.data_segment_bytes_ = 0;
  }

  void ParseHeader() {
    std::memcpy(&header_length_, mapping_, sizeof(header_length_));
    // The on-disk field is explicitly little-endian. Every host this builds
    // for is little-endian too, which the static_assert in fp8_types.hpp
    // effectively pins; a big-endian port would byte-swap here.
    if (header_length_ > file_size_ - kPrefixBytes) {
      throw std::runtime_error(
          "safetensors: '" + path_ + "' declares a " + std::to_string(header_length_) +
          "-byte JSON header but only " + std::to_string(file_size_ - kPrefixBytes) +
          " bytes follow the length prefix. The header itself is incomplete, so "
          "nothing can be read -- this is a corrupt or too-small download, not a "
          "usable slice.");
    }

    data_segment_ = mapping_ + kPrefixBytes + header_length_;
    data_segment_bytes_ = file_size_ - kPrefixBytes - static_cast<std::size_t>(header_length_);

    detail::JsonScanner scanner(reinterpret_cast<const char*>(mapping_ + kPrefixBytes),
                                static_cast<std::size_t>(header_length_));
    scanner.Expect('{');
    if (scanner.TryConsume('}')) {
      return;
    }
    while (true) {
      const std::string key = scanner.ParseString();
      scanner.Expect(':');
      if (key == "__metadata__") {
        ParseMetadata(scanner);
      } else {
        ParseTensorEntry(scanner, key);
      }
      if (scanner.TryConsume('}')) {
        break;
      }
      scanner.Expect(',');
    }
  }

  void ParseMetadata(detail::JsonScanner& scanner) {
    scanner.Expect('{');
    if (scanner.TryConsume('}')) {
      return;
    }
    while (true) {
      const std::string key = scanner.ParseString();
      scanner.Expect(':');
      // The spec says string->string, but tolerate a non-string value by
      // skipping it rather than refusing the whole checkpoint.
      if (scanner.Peek() == '"') {
        metadata_[key] = scanner.ParseString();
      } else {
        scanner.SkipValue();
      }
      if (scanner.TryConsume('}')) {
        return;
      }
      scanner.Expect(',');
    }
  }

  void ParseTensorEntry(detail::JsonScanner& scanner, const std::string& name) {
    TensorView view;
    view.name = name;
    bool saw_dtype = false;
    bool saw_shape = false;
    bool saw_offsets = false;

    scanner.Expect('{');
    if (!scanner.TryConsume('}')) {
      while (true) {
        const std::string field = scanner.ParseString();
        scanner.Expect(':');
        if (field == "dtype") {
          view.dtype = scanner.ParseString();
          saw_dtype = true;
        } else if (field == "shape") {
          view.shape = ParseIntegerArray(scanner);
          saw_shape = true;
        } else if (field == "data_offsets") {
          const std::vector<std::int64_t> offsets = ParseIntegerArray(scanner);
          if (offsets.size() != 2) {
            scanner.Fail("tensor '" + name + "' has " + std::to_string(offsets.size()) +
                         " data_offsets; exactly 2 are required");
          }
          if (offsets[0] < 0 || offsets[1] < offsets[0]) {
            scanner.Fail("tensor '" + name + "' has a non-monotonic data_offsets pair");
          }
          view.data_begin = static_cast<std::uint64_t>(offsets[0]);
          view.data_end = static_cast<std::uint64_t>(offsets[1]);
          saw_offsets = true;
        } else {
          scanner.SkipValue();  // forward compatibility
        }
        if (scanner.TryConsume('}')) {
          break;
        }
        scanner.Expect(',');
      }
    }

    if (!saw_dtype || !saw_shape || !saw_offsets) {
      scanner.Fail("tensor '" + name + "' is missing dtype, shape or data_offsets");
    }

    const std::size_t element_size = SafetensorsDtypeSize(view.dtype);
    if (element_size == 0) {
      throw std::runtime_error("safetensors: tensor '" + name + "' has unknown dtype '" +
                               view.dtype + "'");
    }
    view.byte_size = static_cast<std::size_t>(view.data_end - view.data_begin);

    // The header must be internally consistent even when the data is absent:
    // shape and dtype have to account for exactly the declared byte span.
    const std::int64_t elements = view.elements();
    if (elements < 0) {
      throw std::runtime_error("safetensors: tensor '" + name + "' has a negative extent");
    }
    const std::size_t expected = static_cast<std::size_t>(elements) * element_size;
    if (expected != view.byte_size) {
      throw std::runtime_error(
          "safetensors: tensor '" + name + "' is " + view.dtype + " " +
          view.shape_string() + " = " + std::to_string(expected) +
          " bytes, but data_offsets span " + std::to_string(view.byte_size));
    }

    // Availability: the ONLY place a truncated slice is tolerated.
    if (view.data_end <= data_segment_bytes_) {
      view.available = true;
      view.data_ptr = data_segment_ + view.data_begin;
    } else {
      view.available = false;
      view.data_ptr = nullptr;
    }

    auto inserted = tensors_.emplace(name, std::move(view));
    if (!inserted.second) {
      throw std::runtime_error("safetensors: '" + path_ + "' names tensor '" + name +
                               "' twice");
    }
  }

  static std::vector<std::int64_t> ParseIntegerArray(detail::JsonScanner& scanner) {
    std::vector<std::int64_t> out;
    scanner.Expect('[');
    if (scanner.TryConsume(']')) {
      return out;
    }
    while (true) {
      out.push_back(scanner.ParseInteger());
      if (scanner.TryConsume(']')) {
        return out;
      }
      scanner.Expect(',');
    }
  }

  const std::uint8_t* mapping_ = nullptr;
  std::size_t file_size_ = 0;
  std::uint64_t header_length_ = 0;
  const std::uint8_t* data_segment_ = nullptr;
  std::size_t data_segment_bytes_ = 0;
  std::map<std::string, TensorView> tensors_;
  std::map<std::string, std::string> metadata_;
  std::string path_;
};

}  // namespace moe::emu

#endif  // MOE_EMU_SAFETENSORS_VIEW_HPP
