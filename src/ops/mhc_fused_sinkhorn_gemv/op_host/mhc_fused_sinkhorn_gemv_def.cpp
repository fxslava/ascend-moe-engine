#include "register/op_def_registry.h"

namespace ops {
class MhcFusedSinkhornGemv final : public OpDef {
 public:
  explicit MhcFusedSinkhornGemv(const char* name) : OpDef(name) {
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
    this->Input("phi").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
    this->Input("alpha").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
    this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
    this->Input("h_out").ParamType(REQUIRED).DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
    this->Attr("sinkhorn_iterations").AttrType(OPTIONAL).Int(20);

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicShapeSupportFlag(false)
        .NeedCheckSupportFlag(false)
        .ExtendCfgInfo("opFile.value", "mhc_fused_sinkhorn_gemv_kernel");
    this->AICore().AddConfig("ascend950", config);
  }
};

OP_ADD(MhcFusedSinkhornGemv);
}  // namespace ops
