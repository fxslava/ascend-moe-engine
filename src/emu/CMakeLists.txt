# -----------------------------------------------------------------------------
# moe_emu -- the host CPU emulation of the fused DSV4 decode operators.
#
# STANDALONE BY CONSTRUCTION: this target links nothing but libstdc++/libm. No
# CANN, no libopapi, no libascendcl, no Python, no torch. The root
# CMakeLists.txt returns before its toolkit discovery when ENABLE_HOST_EMU is
# on, so a configure in this mode cannot pick a CANN dependency up even by
# accident -- `dsv4_emu_verify_standalone` checks that claim against the built
# artifacts rather than trusting it.
# -----------------------------------------------------------------------------

add_library(moe_emu STATIC
  # the three fused decode operators
  mhc_fused_sinkhorn_gemv.cpp
  paged_compressor_decode.cpp
  fast_fp8_indexer_topk.cpp
  # the surrounding layer machinery the semi-end-to-end harness threads them
  # through: quantized GEMVs, shared-KV SWA attention, the MoE block
  dsv4_quant_gemv.cpp
  shared_kv_swa_attention.cpp
  moe_block.cpp
  semi_e2e_pipeline.cpp
)

target_include_directories(moe_emu PUBLIC
  "${CMAKE_SOURCE_DIR}/include"
)

target_compile_features(moe_emu PUBLIC cxx_std_17)
target_compile_options(moe_emu PRIVATE ${MOE_EMU_WARNINGS})
target_compile_options(moe_emu PUBLIC ${MOE_EMU_ARCH_FLAGS})

# -ffp-contract=off: without it GCC and Clang fuse `a * b + c` into an FMA at
# their own discretion, and the fused form keeps more intermediate precision.
# That is usually welcome, but this library's whole purpose is to be the golden
# reference an Ascend C kernel is diffed against, so the host result has to be
# the SAME result on every compiler and optimization level. Reproducibility
# beats the last ulp here.
target_compile_options(moe_emu PUBLIC -ffp-contract=off)

if(MOE_EMU_OPENMP)
  target_link_libraries(moe_emu PUBLIC OpenMP::OpenMP_CXX)
endif()
