# aclnnMhcPre

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term> |      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Attention或MLP层的输入矩阵$h^{in}$。

- 计算公式

$$
\begin{aligned}
\vec{x^{'}_{l}} &=RMSNorm(\vec{x_{l}})\\
H^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
H^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
H^{res}_l &= \alpha^{res}_{l} ·(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
H^{pre}_l &= \sigma (H^{pre}_{l})\\
H^{post}_l &= 2\sigma (H^{post}_{l})\\
h_{in} &=\vec{x_{l}}H^{pre}_l
\end{aligned}
$$

---

## 函数原型

算子采用两段式接口调用：需先调用`aclnnMhcPreGetWorkspaceSize`获取计算所需的Device侧内存大小，再调用`aclnnMhcPre`执行实际计算。

```c++
aclnnStatus aclnnMhcPreGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *gammaOptional, double normEps, double hcEps,
    aclTensor *hIn, aclTensor *hPost, aclTensor *hRes,
    aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnMhcPre(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

## aclnnMhcPreGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
| x | 输入 | 待计算数据，表示网络中mHC层的输入数据 | 必选参数，不能为空Tensor | BFLOAT16 或 FLOAT16 | ND | ($B,S,n,D$) 或 ($T,n,D$) | √ |
| phi | 输入 | mHC的参数矩阵 | 必选参数，不能为空Tensor | FLOAT32 | ND | ($n^2+2n, nD$) | √ |
| alpha | 输入 | mHC的缩放参数 | 必选参数，不能为空Tensor | FLOAT32 | - | (3) | - |
| bias | 输入 | mHC的bias参数 | 必选参数，不能为空Tensor | FLOAT32 | - | ($n^2+2n$) | - |
| gammaOptional | 可选输入 | 表示进行RmsNorm计算的缩放因子 | 可选参数 | FLOAT32 | ND | ($n, D$) | √ |
| normEps | 可选输入 | RmsNorm的防除零参数，建议值：1e-6 | 可选参数 | DOUBLE | - | - | - |
| hcEps | 可选输入 | $H_{pre}$的sigmoid后的eps参数，建议值：1e-6 | 可选参数 | DOUBLE | - | - | - |
| hIn | 输出 | 输出的h_in作为Attention/MLP层的输入 | 必选参数 | BFLOAT16 或 FLOAT16  | ND | ($B,S,D$) 或 ($T,D$)  | - |
| hPost | 输出 | 输出的mHC的h_post变换矩阵 | 必选参数 | FLOAT32 | ND | ($B,S,D$) 或 ($T,D$)  | - |
| hRes | 输出 | 输出的mHC的h_res变换矩阵（未做sinkhorn变换） | 必选参数 | FLOAT32 | ND | ($B,S,n,n$) 或 ($T,n,n$) | - |
| invRmsOptional | 可选输出 | RmsNorm计算得到的1/r | 可选参数 | FLOAT32 | ND | ($B,S$) 或 ($T$) | - |
| hMixOptional | 可选输出 | x与phi矩阵乘的结果 | 可选参数 | FLOAT32 | ND | ($B,S,n^2+2n$) 或 ($T,n^2+2n$) | - |
| hPreOptional | 可选输出 | 做完sigmoid计算之后的h_pre矩阵 | 可选参数 | FLOAT32 | ND | ($B,S,n$) 或 ($T,n$) | - |
| workspaceSize | 输出 | 计算所需的Device侧workspace内存大小（字节） | 由算子内部计算得出，用于后续申请内存 | UINT64 | - | - | - |
| executor | 输出 | 算子执行器，包含计算流程和参数信息 | 需传递给第二段接口使用 | aclOpExecutor | - | - | - |

### 返回值

返回`aclnnStatus`状态码，第一段接口主要完成入参校验，异常场景如下：

| 返回值 | 错误码 | 描述 |
|:--- |:--- |:--- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选参数或者输出是空指针。|
| ACLNN_ERR_PARAM_INVALID | 161002 | 输入变量的数据类型和数据格式不在支持的范围内。 |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | API内存调用npu runtime的接口异常。 |

## aclnnMhcPre

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|:--- |:--- |:--- |
| workspace | 输入 | Device侧申请的workspace内存地址，需与第一段接口返回的workspaceSize匹配。 |
| workspaceSize | 输入 | Device侧workspace内存大小，由`aclnnMhcPreGetWorkspaceSize`接口返回。 |
| executor | 输入 | 算子执行器，由第一段接口创建，包含计算流程和参数信息。 |
| stream | 输入 | 指定执行计算任务的Stream，需提前创建并绑定Device。 |

### 返回值

返回`aclnnStatus`状态码。

## 约束说明

### 确定性计算

- aclnnMhcPre 默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束

1. 输入约束：
   - 输入Tensor `x`、`phi`、`alpha`、`bias` 不能为空，且必须为Device侧Tensor；
   - 所有输入/输出Tensor的数据格式仅支持`ACL_FORMAT_ND`；
2. 内存约束：
   - Workspace内存需在Device侧申请，且大小需严格匹配第一段接口返回值；
   - 非连续Tensor无需提前转为连续，算子内部自动处理。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1~65536 | B*S 或T支持512~65536范围（训练及推理Prefill），支持1~512（推理Decode）。|
| n | 4、6、8 | n目前支持4, 6, 8。|
| D | 512~16384 | D支持512~16384范围以内，需满足D为32对齐。|

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

// 计算Tensor形状对应的总元素数
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

// 将Device侧Tensor数据拷贝到Host侧并打印（float类型）
void PrintTensorDataFloat(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

// 将Device侧Tensor数据拷贝到Host侧并打印（float16类型）
void PrintTensorDataFloat16(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<aclFloat16> host_fp16(size);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_fp16.data(), size * sizeof(aclFloat16), device_addr, size * sizeof(aclFloat16),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    for (int64_t i = 0; i < size; ++i) {
        host_data[i] = aclFloat16ToFloat(host_fp16[i]);
    }

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

// 初始化AscendCL环境（Device/Context/Stream）
int InitAcl(int32_t device_id, aclrtContext &context, aclrtStream &stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, error: %d\n", ret); return -1);

    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); return -1);

    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); return -1);

    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); return -1);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); return -1);

    return 0;
}

// 创建FLOAT32类型Device侧aclTensor（含数据拷贝）
int CreateAclTensorFloat32(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                           aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// 创建FLOAT16类型Device侧aclTensor（含数据拷贝）
int CreateAclTensorFloat16(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                           aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<aclFloat16> host_data_fp16(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data_fp16[i] = aclFloat16(host_data[i]);
    }

    int64_t byte_size = size * sizeof(aclFloat16);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, byte_size, host_data_fp16.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// 创建FLOAT16类型输出aclTensor（仅申请内存）
int CreateAclTensorFloat16Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(aclFloat16);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// 创建FLOAT32类型输出aclTensor（仅申请内存）
int CreateAclTensorFloat32Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

struct Tensors {
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr, *gamma_addr = nullptr;
    void *hin_addr = nullptr, *h_post_addr = nullptr, *h_res_addr = nullptr, *inv_rms_addr = nullptr;
    void *h_mix_addr = nullptr, *h_pre_addr = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr, *gamma = nullptr;
    aclTensor *hin = nullptr, *h_post = nullptr, *h_res = nullptr, *inv_rms = nullptr;
    aclTensor *h_mix = nullptr, *h_pre = nullptr;
};

int CreateInputTensors(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &phi_shape,
                       const std::vector<int64_t> &alpha_shape, const std::vector<int64_t> &bias_shape,
                       const std::vector<int64_t> &gamma_shape, Tensors &tensors)
{
    std::vector<float> x_host_data(GetShapeSize(x_shape), 1.0f);
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 1.0f);
    std::vector<float> alpha_host_data(3, 1.0f);
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 1.0f);
    std::vector<float> gamma_host_data(GetShapeSize(gamma_shape), 1.0f);

    int ret = CreateAclTensorFloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(gamma_host_data, gamma_shape, tensors.gamma_addr, tensors.gamma);
    CHECK_RET(ret == 0, LOG_PRINT("Create gamma_tensor failed\n"); return -1);
    return 0;
}

int CreateOutputTensors(const std::vector<int64_t> &hin_shape, const std::vector<int64_t> &h_post_shape,
                        const std::vector<int64_t> &h_res_shape, const std::vector<int64_t> &inv_rms_shape,
                        const std::vector<int64_t> &h_mix_shape, const std::vector<int64_t> &h_pre_shape,
                        Tensors &tensors)
{
    int ret = CreateAclTensorFloat16Output(hin_shape, tensors.hin_addr, tensors.hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_post_shape, tensors.h_post_addr, tensors.h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_res_shape, tensors.h_res_addr, tensors.h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_mix_shape, tensors.h_mix_addr, tensors.h_mix);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_mix_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    return 0;
}

void DestroyTensors(Tensors &tensors)
{
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.gamma);
    aclDestroyTensor(tensors.hin);
    aclDestroyTensor(tensors.h_post);
    aclDestroyTensor(tensors.h_res);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.h_mix);
    aclDestroyTensor(tensors.h_pre);
}

void FreeDeviceMemory(Tensors &tensors)
{
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.gamma_addr);
    aclrtFree(tensors.hin_addr);
    aclrtFree(tensors.h_post_addr);
    aclrtFree(tensors.h_res_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.h_mix_addr);
    aclrtFree(tensors.h_pre_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;
    int B = 1, S = 2048, n = 4, D = 2560;
    std::vector<int64_t> x_shape = {B * S, n, D}, phi_shape = {n * n + 2 * n, n * D}, alpha_shape = {3},
                         bias_shape = {n * n + 2 * n}, gamma_shape = {n, D};
    std::vector<int64_t> hin_shape = {B * S, D}, h_post_shape = {B * S, n}, h_res_shape = {B * S, n, n},
                         inv_rms_shape = {B * S}, h_mix_shape = {B * S, n * n + 2 * n}, h_pre_shape = {B * S, n};
    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);
    ret = CreateInputTensors(x_shape, phi_shape, alpha_shape, bias_shape, gamma_shape, tensors);
    CHECK_RET(ret == 0, return -1);
    ret = CreateOutputTensors(hin_shape, h_post_shape, h_res_shape, inv_rms_shape, h_mix_shape, h_pre_shape, tensors);
    CHECK_RET(ret == 0, return -1);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclnn_ret = aclnnMhcPreGetWorkspaceSize(
        tensors.x, tensors.phi, tensors.alpha, tensors.bias, tensors.gamma, 1e-6, 1e-6, tensors.hin, tensors.h_post,
        tensors.h_res, tensors.inv_rms, tensors.h_mix, tensors.h_pre, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreGetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);
    void *workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }
    aclnn_ret = aclnnMhcPre(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPre failed, error: %d\n", aclnn_ret); return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n"); return -1);
    LOG_PRINT("MhcPre compute success!\nOutput tensor data: \n");
    PrintTensorDataFloat16(hin_shape, tensors.hin_addr);
    PrintTensorDataFloat(h_post_shape, tensors.h_post_addr);
    PrintTensorDataFloat(h_res_shape, tensors.h_res_addr);
    PrintTensorDataFloat(inv_rms_shape, tensors.inv_rms_addr);
    PrintTensorDataFloat(h_mix_shape, tensors.h_mix_addr);
    PrintTensorDataFloat(h_pre_shape, tensors.h_pre_addr);
    DestroyTensors(tensors);
    FreeDeviceMemory(tensors);
    if (workspace_size > 0)
        aclrtFree(workspace_addr);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();
    LOG_PRINT("All resources released successfully!\n");
    return 0;
}
```
