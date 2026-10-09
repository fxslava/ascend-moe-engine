/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

// Stride derivation. See kv_cache_layout.hpp for why no call site is allowed
// to compute one of these by hand.

#include "moe/core/kv_cache_layout.hpp"

#include <aclnn/acl_meta.h>

#include <memory>
#include <sstream>

#include "moe/core/config.hpp"
#include "moe/core/error.hpp"

namespace ascend_moe {

size_t AclDataTypeBytes(int32_t acl_dtype) {
  switch (acl_dtype) {
    case kAclUint8:
    case kAclFloat8E4m3Fn:
    case kAclFloat8E8m0:
    // Two E2M1 nibbles share a byte; CreateFp4Tensor's callers halve the
    // element count, so the per-element size reported here is 1.
    case kAclFloat4E2m1:
      return 1;
    case kAclFloat16:
    case kAclBf16:
      return 2;
    case kAclFloat32:
    case kAclInt32:
      return 4;
    case kAclInt64:
      return 8;
    default:
      throw Dsv4Error("no byte width is known for aclDataType code " + std::to_string(acl_dtype) +
                      "; add it to AclDataTypeBytes rather than assuming a size");
  }
}

std::string TensorSliceLayout::Describe() const {
  std::ostringstream text;
  text << "dims [";
  for (size_t index = 0; index < dims.size(); ++index) {
    text << (index == 0 ? "" : ", ") << dims[index];
  }
  text << "] strides [";
  for (size_t index = 0; index < strides.size(); ++index) {
    text << (index == 0 ? "" : ", ") << strides[index];
  }
  text << "] x " << element_bytes << "B";
  return text.str();
}

TensorSliceLayout ReadTensorSliceLayout(const aclTensor* tensor) {
  DSV4_REQUIRE(tensor != nullptr, "a stride cannot be derived from a null descriptor");
  TensorSliceLayout layout;

  int64_t* raw = nullptr;
  uint64_t count = 0;
  DSV4_ACL_CHECK(aclGetViewShape(tensor, &raw, &count));
  {
    const std::unique_ptr<int64_t[]> owned(raw);
    layout.dims.assign(owned.get(), owned.get() + count);
  }
  raw = nullptr;
  count = 0;
  DSV4_ACL_CHECK(aclGetViewStrides(tensor, &raw, &count));
  {
    const std::unique_ptr<int64_t[]> owned(raw);
    layout.strides.assign(owned.get(), owned.get() + count);
  }
  aclDataType dtype{};
  DSV4_ACL_CHECK(aclGetDataType(tensor, &dtype));
  layout.dtype = static_cast<int32_t>(dtype);
  layout.element_bytes = AclDataTypeBytes(layout.dtype);

  DSV4_REQUIRE(!layout.dims.empty(), "a descriptor with no dimensions has no stride to derive");
  DSV4_REQUIRE(layout.strides.size() == layout.dims.size(),
               "the descriptor reports " << layout.dims.size() << " dimensions but "
                                         << layout.strides.size() << " strides");
  return layout;
}

int64_t DeriveDimensionStrideElements(const aclTensor* tensor, size_t axis) {
  const TensorSliceLayout layout = ReadTensorSliceLayout(tensor);
  DSV4_REQUIRE(axis < layout.rank(), "axis " << axis << " is outside the rank-" << layout.rank()
                                             << " descriptor (" << layout.Describe() << ")");
  return layout.strides[axis];
}

size_t DeriveDimensionStrideBytes(const aclTensor* tensor, size_t axis) {
  const TensorSliceLayout layout = ReadTensorSliceLayout(tensor);
  DSV4_REQUIRE(axis < layout.rank(), "axis " << axis << " is outside the rank-" << layout.rank()
                                             << " descriptor (" << layout.Describe() << ")");
  DSV4_REQUIRE(layout.strides[axis] >= 0,
               "axis " << axis << " has a negative element stride; a byte stride would be meaningless ("
                       << layout.Describe() << ")");
  return static_cast<size_t>(layout.strides[axis]) * layout.element_bytes;
}

void AssertDimensionStride(const aclTensor* tensor, size_t axis, int64_t expected, const char* what) {
  const int64_t actual = DeriveDimensionStrideElements(tensor, axis);
  DSV4_REQUIRE(actual == expected,
               what << ": axis-" << axis << " element stride is " << actual << ", but " << expected
                    << " was derived for the attribute. The vendored wrappers forward this value to the "
                       "kernel unchecked, so a disagreement would address the wrong block silently.");
}

}  // namespace ascend_moe
