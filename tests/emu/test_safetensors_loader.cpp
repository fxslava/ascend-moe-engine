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

// safetensors_view.hpp: header parsing, dtype and shape extraction, zero-copy
// pointers, and the truncated-slice behaviour the Layer 0 bench depends on.
//
// Every fixture is written here byte by byte, so the test knows the exact
// offsets it is asserting rather than trusting a writer library. When the real
// DeepSeek-V4-Flash shard is present (data/dsv4_layer0/), the last section
// also parses it and checks the layer-0 geometry against config.json.

#include "moe/emu/safetensors_view.hpp"
#include "moe/emu/cpu_kernels.h"
#include "moe/emu/fp8_types.hpp"

#include "emu_test_support.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace moe::emu;

// --- a byte-exact fixture writer --------------------------------------------

struct Entry {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

std::string TempPath(const std::string& stem) {
  const char* base = std::getenv("TMPDIR");
  std::string directory = (base == nullptr) ? "/tmp" : base;
  return directory + "/emu_safetensors_" + stem + "_" +
         std::to_string(static_cast<long>(::getpid())) + ".safetensors";
}

// Builds the file and returns its path. `truncate_to` (when non-zero) cuts the
// file short, simulating the leading-window download.
std::string WriteFixture(const std::string& stem, const std::vector<Entry>& entries,
                         const std::string& metadata_json = "",
                         std::size_t truncate_to = 0) {
  std::string header = "{";
  if (!metadata_json.empty()) {
    header += "\"__metadata__\":" + metadata_json + ",";
  }
  std::uint64_t offset = 0;
  std::vector<std::uint8_t> data;
  bool first = true;
  for (const Entry& entry : entries) {
    if (!first) {
      header += ",";
    }
    first = false;
    header += "\"" + entry.name + "\":{\"dtype\":\"" + entry.dtype + "\",\"shape\":[";
    for (std::size_t i = 0; i < entry.shape.size(); ++i) {
      if (i != 0) {
        header += ",";
      }
      header += std::to_string(entry.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + entry.bytes.size()) + "]}";
    data.insert(data.end(), entry.bytes.begin(), entry.bytes.end());
    offset += entry.bytes.size();
  }
  header += "}";

  const std::uint64_t header_length = header.size();
  std::vector<std::uint8_t> file(8);
  std::memcpy(file.data(), &header_length, sizeof(header_length));
  file.insert(file.end(), header.begin(), header.end());
  file.insert(file.end(), data.begin(), data.end());
  if (truncate_to != 0 && truncate_to < file.size()) {
    file.resize(truncate_to);
  }

  const std::string path = TempPath(stem);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(file.data()),
            static_cast<std::streamsize>(file.size()));
  out.close();
  return path;
}

template <typename T>
std::vector<std::uint8_t> Raw(const std::vector<T>& values) {
  std::vector<std::uint8_t> out(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(out.data(), values.data(), out.size());
  }
  return out;
}

// --- tests ------------------------------------------------------------------

void TestDtypeSizes(emu_test::Runner& runner) {
  runner.Section("dtype table");
  runner.Check(SafetensorsDtypeSize("F32") == 4, "F32 is 4 bytes");
  runner.Check(SafetensorsDtypeSize("F64") == 8, "F64 is 8 bytes");
  runner.Check(SafetensorsDtypeSize("BF16") == 2, "BF16 is 2 bytes");
  runner.Check(SafetensorsDtypeSize("F16") == 2, "F16 is 2 bytes");
  runner.Check(SafetensorsDtypeSize("I64") == 8, "I64 is 8 bytes");
  runner.Check(SafetensorsDtypeSize("I32") == 4, "I32 is 4 bytes");
  runner.Check(SafetensorsDtypeSize("BOOL") == 1, "BOOL is 1 byte");
  // The three FP8 spellings a DSV4 checkpoint actually carries.
  runner.Check(SafetensorsDtypeSize("F8_E4M3") == 1, "F8_E4M3 is 1 byte");
  runner.Check(SafetensorsDtypeSize("F8_E5M2") == 1, "F8_E5M2 is 1 byte");
  runner.Check(SafetensorsDtypeSize("F8_E8M0") == 1, "F8_E8M0 is 1 byte");
  runner.Check(SafetensorsDtypeSize("NOT_A_DTYPE") == 0,
               "an unknown dtype reports 0 rather than guessing a width");
}

void TestBasicParse(emu_test::Runner& runner) {
  runner.Section("offsets, dtypes and shapes on a mock file");

  const std::vector<float> phi_values = {1.0F, -2.5F, 3.25F, 4.0F, 0.5F, -0.125F};
  const std::vector<std::uint16_t> bf16_values = {
      FloatToBf16(1.0F), FloatToBf16(-1.0F), FloatToBf16(0.25F), FloatToBf16(128.0F)};
  const std::vector<std::uint8_t> fp8_values = {
      EncodeFp8E4M3(1.0F), EncodeFp8E4M3(-1.0F), EncodeFp8E4M3(448.0F),
      EncodeFp8E4M3(0.015625F)};
  const std::vector<std::uint8_t> e8m0_values = {127, 128, 126, 0};
  const std::vector<std::int64_t> i64_values = {-1, 0, 1, 1LL << 40};

  const std::vector<Entry> entries = {
      {"phi", "F32", {2, 3}, Raw(phi_values)},
      {"gain", "BF16", {4}, Raw(bf16_values)},
      {"quant.weight", "F8_E4M3", {2, 2}, fp8_values},
      {"quant.scale", "F8_E8M0", {4}, e8m0_values},
      {"routing", "I64", {2, 2}, Raw(i64_values)},
      {"scalar", "F32", {}, Raw(std::vector<float>{42.0F})},
  };
  const std::string path =
      WriteFixture("basic", entries, R"({"format":"pt","note":"a\"quoted\" value"})");

  SafetensorsView view(path);
  runner.Check(view.is_open(), "the file opened");
  runner.Check(view.size() == entries.size(),
               "all " + std::to_string(entries.size()) + " tensors were parsed");

  // __metadata__ is not a tensor and its escapes decode.
  runner.Check(!view.contains("__metadata__"),
               "__metadata__ is not reported as a tensor");
  runner.Check(view.metadata().count("format") == 1 &&
                   view.metadata().at("format") == "pt",
               "__metadata__.format == pt");
  runner.Check(view.metadata().count("note") == 1 &&
                   view.metadata().at("note") == "a\"quoted\" value",
               "a \\\" escape inside __metadata__ decodes");

  // --- F32, including the exact byte offsets --------------------------------
  {
    const TensorView& phi = view.Tensor("phi");
    runner.Check(phi.dtype == "F32", "phi dtype is F32");
    runner.Check(phi.is_shape({2, 3}), "phi shape is [2, 3]");
    runner.Check(phi.shape_string() == "[2, 3]", "phi shape prints as [2, 3]");
    runner.Check(phi.elements() == 6, "phi has 6 elements");
    runner.Check(phi.element_size() == 4, "phi element size is 4");
    runner.Check(phi.byte_size == 24, "phi occupies 24 bytes");
    runner.Check(phi.data_begin == 0 && phi.data_end == 24,
                 "phi spans data bytes [0, 24)");
    runner.Check(phi.available, "phi is available");
    for (std::size_t i = 0; i < phi_values.size(); ++i) {
      runner.CheckNear(phi.ElementAt<float>(i), phi_values[i], 0.0,
                       "phi element " + std::to_string(i));
    }
    std::vector<float> copied(phi_values.size());
    phi.CopyTo(copied.data(), copied.size());
    runner.Check(copied == phi_values, "CopyTo reproduces phi exactly");
  }

  // --- BF16, at a non-zero offset ------------------------------------------
  {
    const TensorView& gain = view.Tensor("gain");
    runner.Check(gain.data_begin == 24, "gain starts right after phi, at 24");
    runner.Check(gain.byte_size == 8, "gain occupies 8 bytes");
    for (std::size_t i = 0; i < bf16_values.size(); ++i) {
      runner.Check(gain.ElementAt<std::uint16_t>(i) == bf16_values[i],
                   "gain BF16 word " + std::to_string(i));
    }
    runner.CheckNear(Bf16ToFloat(gain.ElementAt<std::uint16_t>(3)), 128.0, 0.0,
                     "gain[3] widens to 128.0");
  }

  // --- FP8 ------------------------------------------------------------------
  {
    const TensorView& weight = view.Tensor("quant.weight");
    runner.Check(weight.dtype == "F8_E4M3", "quant.weight dtype is F8_E4M3");
    runner.Check(weight.is_shape({2, 2}), "quant.weight shape is [2, 2]");
    runner.Check(weight.byte_size == 4, "quant.weight occupies 4 bytes");
    runner.CheckNear(DecodeFp8E4M3(weight.ElementAt<std::uint8_t>(0)), 1.0, 0.0,
                     "quant.weight[0] decodes to 1.0");
    runner.CheckNear(DecodeFp8E4M3(weight.ElementAt<std::uint8_t>(2)), 448.0, 0.0,
                     "quant.weight[2] decodes to 448.0");

    const TensorView& scale = view.Tensor("quant.scale");
    runner.CheckNear(DecodeFp8E8M0(scale.ElementAt<std::uint8_t>(0)), 1.0, 0.0,
                     "quant.scale[0] is 2^0");
    runner.CheckNear(DecodeFp8E8M0(scale.ElementAt<std::uint8_t>(1)), 2.0, 0.0,
                     "quant.scale[1] is 2^1");
  }

  // --- I64 and a rank-0 scalar ---------------------------------------------
  {
    const TensorView& routing = view.Tensor("routing");
    runner.Check(routing.ElementAt<std::int64_t>(3) == (1LL << 40),
                 "a 2^40 I64 value survives");
    const TensorView& scalar = view.Tensor("scalar");
    runner.Check(scalar.shape.empty(), "a rank-0 tensor has an empty shape");
    runner.Check(scalar.elements() == 1, "a rank-0 tensor has 1 element");
    runner.CheckNear(scalar.ElementAt<float>(0), 42.0, 0.0, "the scalar is 42");
  }

  // --- zero copy ------------------------------------------------------------
  {
    const TensorView& phi = view.Tensor("phi");
    const TensorView& gain = view.Tensor("gain");
    const auto* phi_bytes = static_cast<const std::uint8_t*>(phi.data_ptr);
    const auto* gain_bytes = static_cast<const std::uint8_t*>(gain.data_ptr);
    runner.Check(gain_bytes == phi_bytes + phi.byte_size,
                 "the views point into one contiguous mapping (zero copy)");
  }

  // --- lookups --------------------------------------------------------------
  runner.Check(view.Find("nope") == nullptr, "Find returns null for an absent name");
  runner.Check(view.contains("phi") && !view.contains("nope"), "contains() works");
  {
    const std::vector<const TensorView*> quant = view.WithPrefix("quant.");
    runner.Check(quant.size() == 2, "WithPrefix(\"quant.\") finds both tensors");
    const std::vector<std::string> all = view.names();
    runner.Check(all.size() == entries.size(), "names() lists every tensor");
  }

  bool threw = false;
  try {
    view.Tensor("absent");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  runner.Check(threw, "Tensor() throws on an absent name");

  threw = false;
  try {
    view.TensorWithShape("phi", {3, 2});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  runner.Check(threw, "TensorWithShape rejects a transposed shape");
  runner.Check(view.TensorWithShape("phi", {2, 3}).elements() == 6,
               "TensorWithShape accepts the right shape");

  threw = false;
  try {
    view.Tensor("phi").ElementAt<std::uint16_t>(0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  runner.Check(threw, "reading an F32 tensor as 2-byte elements is refused");

  threw = false;
  try {
    view.Tensor("phi").ElementAt<float>(6);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  runner.Check(threw, "reading past the last element is refused");

  ::unlink(path.c_str());
}

void TestMisalignedDataSegment(emu_test::Runner& runner) {
  runner.Section("a data segment that is not 4-byte aligned");
  // The data segment starts at 8 + header_len. By padding the metadata with
  // one extra character at a time, some fixture WILL place an F32 tensor at an
  // odd offset -- which is exactly the case where casting data_ptr to
  // `const float*` is undefined behaviour and ElementAt's memcpy is not.
  int misaligned_cases = 0;
  for (int pad = 0; pad < 4; ++pad) {
    const std::vector<float> values = {1.5F, -2.5F, 3.5F};
    const std::string metadata =
        "{\"pad\":\"" + std::string(static_cast<std::size_t>(pad), 'x') + "\"}";
    const std::string path =
        WriteFixture("align" + std::to_string(pad),
                     {{"t", "F32", {3}, Raw(values)}}, metadata);
    SafetensorsView view(path);
    const TensorView& t = view.Tensor("t");
    const auto address = reinterpret_cast<std::uintptr_t>(t.data_ptr);
    if (address % alignof(float) != 0) {
      ++misaligned_cases;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
      runner.CheckNear(t.ElementAt<float>(i), values[i], 0.0,
                       "pad " + std::to_string(pad) + " element " + std::to_string(i));
    }
    ::unlink(path.c_str());
  }
  runner.Check(misaligned_cases > 0,
               "at least one fixture really did land misaligned, so the memcpy "
               "path was exercised");
}

void TestTruncatedSlice(emu_test::Runner& runner) {
  runner.Section("a truncated leading slice: complete header, partial data");

  const std::vector<float> small = {1.0F, 2.0F};
  const std::vector<float> large(1000, 7.0F);
  const std::vector<Entry> entries = {
      {"early", "F32", {2}, Raw(small)},
      {"late", "F32", {1000}, Raw(large)},
  };

  // Keep the header plus `early`'s 8 bytes and nothing of `late`.
  const std::string full_path = WriteFixture("fullref", entries);
  std::size_t header_bytes = 0;
  {
    SafetensorsView reference(full_path);
    header_bytes = 8 + static_cast<std::size_t>(reference.header_length());
  }
  const std::string path = WriteFixture("slice", entries, "", header_bytes + 8);

  SafetensorsView view(path);
  runner.Check(view.size() == 2,
               "the header still describes BOTH tensors in a truncated file");
  runner.Check(view.data_segment_bytes() == 8,
               "only 8 bytes of the data segment are present");

  const TensorView* early = view.Find("early");
  runner.Check(early != nullptr && early->available,
               "the tensor inside the window is available");
  runner.CheckNear(early->ElementAt<float>(1), 2.0, 0.0, "and reads correctly");

  const TensorView* late = view.Find("late");
  runner.Check(late != nullptr, "Find still returns the tensor outside the window");
  runner.Check(!late->available, "but it is marked unavailable");
  runner.Check(late->data_ptr == nullptr, "and carries no pointer to fault on");
  runner.Check(late->dtype == "F32" && late->is_shape({1000}),
               "its dtype and shape are still known from the header");
  runner.Check(late->byte_size == 4000,
               "and so is the byte size it WOULD have occupied");

  bool threw = false;
  std::string message;
  try {
    view.Tensor("late");
  } catch (const std::runtime_error& error) {
    threw = true;
    message = error.what();
  }
  runner.Check(threw, "Tensor() refuses an unavailable tensor");
  runner.Check(message.find("truncated") != std::string::npos,
               "and the message says the file is a truncated slice");
  runner.Check(message.find("--full") != std::string::npos,
               "and names the flag that fixes it");

  ::unlink(path.c_str());
  ::unlink(full_path.c_str());
}

void TestMalformedFiles(emu_test::Runner& runner) {
  runner.Section("malformed input is rejected, never read past");

  const auto expect_throw = [&](const std::string& stem,
                                const std::vector<std::uint8_t>& bytes,
                                const std::string& what) {
    const std::string path = TempPath(stem);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    bool threw = false;
    try {
      SafetensorsView view(path);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    runner.Check(threw, what);
    ::unlink(path.c_str());
  };

  const auto with_header = [](const std::string& header) {
    const std::uint64_t length = header.size();
    std::vector<std::uint8_t> bytes(8);
    std::memcpy(bytes.data(), &length, sizeof(length));
    bytes.insert(bytes.end(), header.begin(), header.end());
    return bytes;
  };

  expect_throw("tooshort", {1, 2, 3}, "a file shorter than 8 bytes is rejected");

  // A header length that runs past EOF: the HEADER itself is incomplete, which
  // is corruption rather than a usable slice.
  {
    std::vector<std::uint8_t> bytes(8);
    const std::uint64_t length = 1000000;
    std::memcpy(bytes.data(), &length, sizeof(length));
    bytes.push_back('{');
    expect_throw("hdrpast", bytes, "a header_len past EOF is rejected");
  }

  expect_throw("notjson", with_header("this is not json"),
               "a non-JSON header is rejected");
  expect_throw("unterm", with_header("{\"a\":{\"dtype\":\"F32\""),
               "an unterminated object is rejected");
  expect_throw("nodtype", with_header("{\"a\":{\"shape\":[1],\"data_offsets\":[0,4]}}"),
               "a tensor with no dtype is rejected");
  expect_throw("noshape", with_header("{\"a\":{\"dtype\":\"F32\",\"data_offsets\":[0,4]}}"),
               "a tensor with no shape is rejected");
  expect_throw("nooffs", with_header("{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}"),
               "a tensor with no data_offsets is rejected");
  expect_throw("baddtype",
               with_header("{\"a\":{\"dtype\":\"F7_E3M3\",\"shape\":[1],"
                           "\"data_offsets\":[0,4]}}"),
               "an unknown dtype is rejected");
  // shape * element_size must equal the declared byte span: the check that
  // stops a wrong-stride read.
  expect_throw("sizemismatch",
               with_header("{\"a\":{\"dtype\":\"F32\",\"shape\":[4],"
                           "\"data_offsets\":[0,8]}}"),
               "shape/dtype disagreeing with data_offsets is rejected");
  expect_throw("backwards",
               with_header("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],"
                           "\"data_offsets\":[8,4]}}"),
               "a reversed data_offsets pair is rejected");
  expect_throw("negoffset",
               with_header("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],"
                           "\"data_offsets\":[-4,0]}}"),
               "a negative data offset is rejected");
  expect_throw("threeoffs",
               with_header("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],"
                           "\"data_offsets\":[0,4,8]}}"),
               "three data_offsets are rejected");
  expect_throw("floatshape",
               with_header("{\"a\":{\"dtype\":\"F32\",\"shape\":[1.5],"
                           "\"data_offsets\":[0,4]}}"),
               "a fractional shape extent is rejected");

  // Unknown fields must be SKIPPED, not rejected: the format may grow.
  {
    const std::string header =
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],"
        "\"future\":{\"nested\":[1,2,{\"deep\":true}]},\"flag\":false,"
        "\"nothing\":null,\"ratio\":1.5e-3}}";
    std::vector<std::uint8_t> bytes = with_header(header);
    const std::vector<float> value = {3.0F};
    const std::vector<std::uint8_t> raw = Raw(value);
    bytes.insert(bytes.end(), raw.begin(), raw.end());
    const std::string path = TempPath("future");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    bool ok = false;
    try {
      SafetensorsView view(path);
      ok = view.size() == 1 && view.Tensor("a").ElementAt<float>(0) == 3.0F;
    } catch (const std::runtime_error&) {
      ok = false;
    }
    runner.Check(ok, "unknown header fields are skipped, not rejected");
    ::unlink(path.c_str());
  }

  // An empty header object is legal: a shard with no tensors.
  {
    const std::vector<std::uint8_t> bytes = with_header("{}");
    const std::string path = TempPath("empty");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    bool ok = false;
    try {
      SafetensorsView view(path);
      ok = view.size() == 0;
    } catch (const std::runtime_error&) {
      ok = false;
    }
    runner.Check(ok, "an empty header object parses to zero tensors");
    ::unlink(path.c_str());
  }

  bool threw = false;
  try {
    SafetensorsView view("/nonexistent/path/to/nothing.safetensors");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  runner.Check(threw, "a missing file is reported");
}

void TestMoveSemantics(emu_test::Runner& runner) {
  runner.Section("move semantics keep the mapping alive");
  const std::vector<float> values = {9.0F, 8.0F};
  const std::string path = WriteFixture("move", {{"t", "F32", {2}, Raw(values)}});
  SafetensorsView first(path);
  const void* original = first.Tensor("t").data_ptr;
  SafetensorsView second(std::move(first));
  runner.Check(!first.is_open(), "the moved-from view is closed");
  runner.Check(second.is_open(), "the moved-to view is open");
  runner.Check(second.Tensor("t").data_ptr == original,
               "the data pointer is unchanged by the move");
  runner.CheckNear(second.Tensor("t").ElementAt<float>(0), 9.0, 0.0,
                   "and still reads correctly");
  ::unlink(path.c_str());
}

// --- the real checkpoint, when it has been downloaded -----------------------

void TestRealShardIfPresent(emu_test::Runner& runner) {
  runner.Section("the real DeepSeek-V4-Flash shard, if downloaded");
  const char* override_path = std::getenv("DSV4_LAYER0_SHARD");
  const std::string path =
      (override_path != nullptr)
          ? std::string(override_path)
          : std::string("data/dsv4_layer0/model-00002-of-00046.safetensors");

  std::ifstream probe(path, std::ios::binary);
  if (!probe.good()) {
    std::printf("   SKIP: %s is not present (run scripts/download_dsv4_layer0.sh)\n",
                path.c_str());
    return;
  }
  probe.close();

  SafetensorsView view(path);
  std::printf("   %s: %zu tensors, header %llu bytes, %zu data bytes mapped\n",
              path.c_str(), view.size(),
              static_cast<unsigned long long>(view.header_length()),
              view.data_segment_bytes());

  // The six mHC tensors the Layer 0 pipeline needs, at the shapes
  // config.json implies: hc_mult 4 and hidden_size 4096 give
  // phi [n^2 + 2n, n * D] = [24, 16384].
  for (const char* stem : {"attn", "ffn"}) {
    const std::string prefix = std::string("layers.0.hc_") + stem;
    const TensorView& phi = view.TensorWithShape(prefix + "_fn", {kHcPhiRows, kHcPhiCols});
    runner.Check(phi.dtype == "F32", prefix + "_fn is F32");
    const TensorView& alpha =
        view.TensorWithShape(prefix + "_scale", {kHcAlphaElements});
    runner.Check(alpha.dtype == "F32", prefix + "_scale is F32");
    const TensorView& bias =
        view.TensorWithShape(prefix + "_base", {kHcProjectionRows});
    runner.Check(bias.dtype == "F32", prefix + "_base is F32");

    // The weights must be finite and not all zero -- the cheapest proof that
    // the offsets point at real data and not at a hole in the slice.
    bool finite = true;
    bool nonzero = false;
    for (std::int64_t i = 0; i < 4096; ++i) {
      const float value = phi.ElementAt<float>(static_cast<std::size_t>(i));
      if (!std::isfinite(value)) {
        finite = false;
      }
      if (value != 0.0F) {
        nonzero = true;
      }
    }
    runner.Check(finite, prefix + "_fn holds only finite values");
    runner.Check(nonzero, prefix + "_fn is not all zeros");
  }

  // The norm weights, at hidden_size / the MLA ranks config.json publishes.
  runner.Check(view.TensorWithShape("layers.0.attn_norm.weight", {kHiddenSize}).dtype ==
                   "BF16",
               "layers.0.attn_norm.weight is BF16 [4096]");
  runner.Check(view.TensorWithShape("layers.0.ffn_norm.weight", {kHiddenSize}).dtype ==
                   "BF16",
               "layers.0.ffn_norm.weight is BF16 [4096]");
  runner.Check(view.TensorWithShape("layers.0.attn.kv_norm.weight", {512}).dtype ==
                   "BF16",
               "layers.0.attn.kv_norm.weight is BF16 [512] (kv_lora_rank)");

  // THE FP8 PROJECTIONS ARE PAST THE DEFAULT 16 MiB WINDOW -- and that is the
  // whole point of the availability flag. In a slice build they must be
  // present in the header and absent in the data.
  const TensorView* wkv = view.Find("layers.0.attn.wkv.weight");
  runner.Check(wkv != nullptr, "layers.0.attn.wkv.weight is in the header");
  if (wkv != nullptr) {
    runner.Check(wkv->dtype == "F8_E4M3", "and it is F8_E4M3");
    runner.Check(wkv->is_shape({512, 4096}), "and shaped [512, 4096]");
    if (!wkv->available) {
      std::printf("   (as expected for a sliced shard, the FP8 projections are "
                  "outside the window)\n");
    }
  }
  // Its microscale block grid must match "weight_block_size": [128, 128]:
  // 512/128 = 4 by 4096/128 = 32.
  const TensorView* scale = view.Find("layers.0.attn.wkv.scale");
  runner.Check(scale != nullptr && scale->dtype == "F8_E8M0" &&
                   scale->is_shape({4, 32}),
               "layers.0.attn.wkv.scale is F8_E8M0 [4, 32] = a 128x128 block grid");

  // Layer 0 carries NO compressor and NO indexer: config.json's
  // compress_ratios[0] is 0, i.e. an uncompressed (SWA) layer.
  runner.Check(view.WithPrefix("layers.0.attn.compressor.").empty(),
               "layer 0 has no compressor tensors (compress_ratios[0] == 0)");
  runner.Check(view.WithPrefix("layers.0.attn.indexer.").empty(),
               "layer 0 has no indexer tensors");
}

}  // namespace

int main() {
  emu_test::Runner runner("test_safetensors_loader");
  try {
    TestDtypeSizes(runner);
    TestBasicParse(runner);
    TestMisalignedDataSegment(runner);
    TestTruncatedSlice(runner);
    TestMalformedFiles(runner);
    TestMoveSemantics(runner);
    TestRealShardIfPresent(runner);
  } catch (const std::exception& error) {
    std::printf("   FATAL: unexpected exception: %s\n", error.what());
    return 1;
  }
  return runner.Summary();
}
