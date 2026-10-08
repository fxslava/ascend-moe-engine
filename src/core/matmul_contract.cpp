#include <array>
#include <memory>
#include <sstream>

#include "moe/core/op_table.hpp"

namespace ascend_moe {
void ValidateDenseMatmul(const aclTensor* self, const aclTensor* mat2, aclTensor* out,
                         int8_t cube_math_type) {
  struct Matrix {
    std::array<int64_t, 2> shape;
    aclDataType dtype;
  };
  auto inspect = [](const aclTensor* tensor, const char* name) -> Matrix {
    DSV4_REQUIRE(tensor, "aclnnMatmul: " << name << " is null (161001)");
    int64_t* raw = nullptr;
    uint64_t count = 0;
    DSV4_ACL_CHECK(aclGetViewShape(tensor, &raw, &count));
    std::unique_ptr<int64_t[]> dims(raw);
    DSV4_REQUIRE(count == 2 && dims[0] > 0 && dims[1] > 0,
                 "aclnnMatmul: " << name << " must be a nonempty 2-D matrix");
    DSV4_ACL_CHECK(aclGetViewStrides(tensor, &raw, &count));
    std::unique_ptr<int64_t[]> strides(raw);
    DSV4_REQUIRE(count == 2 && strides[0] == dims[1] && strides[1] == 1,
                 "aclnnMatmul: " << name << " must have contiguous row-major strides");
    aclDataType dtype{};
    aclFormat format{};
    void* address = nullptr;
    int64_t offset = 0;
    DSV4_ACL_CHECK(aclGetDataType(tensor, &dtype));
    DSV4_ACL_CHECK(aclGetFormat(tensor, &format));
    DSV4_ACL_CHECK(aclGetRawTensorAddr(tensor, &address));
    DSV4_ACL_CHECK(aclGetViewOffset(tensor, &offset));
    DSV4_REQUIRE(
        format == ACL_FORMAT_ND && offset == 0 && address && reinterpret_cast<uintptr_t>(address) % 32 == 0,
        "aclnnMatmul: " << name << " requires ND, zero offset and a 32-byte-aligned address");
    DSV4_REQUIRE(
        dtype == ACL_BF16 || dtype == ACL_FLOAT16,
        "aclnnMatmul: "
            << name << " dtype=" << static_cast<int>(dtype)
            << "; dense path requires BF16/FP16; quantized weights require QuantMatmulV5/GroupedMatmulV5");
    return {{dims[0], dims[1]}, dtype};
  };
  const auto a = inspect(self, "A");
  const auto b = inspect(mat2, "B");
  const auto c = inspect(out, "C");
  DSV4_REQUIRE(a.dtype == b.dtype && a.dtype == c.dtype,
               "aclnnMatmul: A/B/C dtypes must match; cast explicitly before planning");
  DSV4_REQUIRE(a.shape[1] == b.shape[0] && c.shape[0] == a.shape[0] && c.shape[1] == b.shape[1],
               "aclnnMatmul: A=[" << a.shape[0] << ',' << a.shape[1] << "], B=[" << b.shape[0] << ','
                                  << b.shape[1] << "], C=[" << c.shape[0] << ',' << c.shape[1]
                                  << "]; expected [M,K] x [K,N] -> [M,N], no implicit weight transpose");
  DSV4_REQUIRE(cube_math_type == 0, "dense Matmul requires KEEP_DTYPE (cubeMathType=0)");
}
}  // namespace ascend_moe
