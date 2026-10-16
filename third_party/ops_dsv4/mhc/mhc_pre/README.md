# MhcPre

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
h_{in} &=\vec{x^{'}_{l}}H^{pre}_l
\end{aligned}
$$

---

## 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
| x | 输入 | 待计算数据，表示网络中mHC层的输入数据 | 必选参数，不能为空Tensor | BFLOAT16 或 FLOAT16 | ND | ($B,S,n,D$) 或 ($T,n,D$) | √ |
| phi | 输入 | mHC的参数矩阵 | 必选参数，不能为空Tensor | FLOAT32 | ND | ($n^2+2n, nD$) | √ |
| alpha | 输入 | mHC的缩放参数 | 必选参数，不能为空Tensor | FLOAT32 | - | (3) | - |
| bias | 输入 | mHC的bias参数 | 必选参数，不能为空Tensor | FLOAT32 | - | ($n^2+2n$) | - |
 | gammaOptional | 可选输入 | 表示进行RmsNorm计算的缩放因子 | 可选参数 | FLOAT32 | ND | ($n, D$) | √ |
 | normEps | 可选输入 | RmsNorm的防除零参数 | 可选参数 | DOUBLE | - | - | - |
 | hcEps | 可选输入 | $H_{pre}$的sigmoid后的eps参数 | 可选参数 | DOUBLE | - | - | - |
| hIn | 输出 | 输出的h_in作为Attention/MLP层的输入 | 必选参数 | BFLOAT16 或 FLOAT16  | ND | ($B,S,D$) 或 ($T,D$)  | - |
| hPost | 输出 | 输出的mHC的h_post变换矩阵 | 必选参数 | FLOAT32 | ND | ($B,S,D$) 或 ($T,D$)  | - |
| hRes | 输出 | 输出的mHC的h_res变换矩阵（未做sinkhorn变换） | 必选参数 | FLOAT32 | ND | ($B,S,n,n$) 或 ($T,n,n$) | - |
| invRmsOptional | 可选输出 | RmsNorm计算得到的1/r | 可选参数 | FLOAT32 | ND | ($B,S$) 或 ($T$) | - |
| hMixOptional | 可选输出 | x与phi矩阵乘的结果 | 可选参数 | FLOAT32 | ND | ($B,S,n^2+2n$) 或 ($T,n^2+2n$) | - |
| hPreOptional | 可选输出 | 做完sigmoid计算之后的h_pre矩阵 | 可选参数 | FLOAT32 | ND | ($B,S,n$) 或 ($T,n$) | - |

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

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_pre](examples/test_aclnn_mhc_pre.cpp) | 通过接口方式调用[aclnnMhcPre](docs/aclnnMhcPre.md)算子。 |
