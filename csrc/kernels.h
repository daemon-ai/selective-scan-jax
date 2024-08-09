#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace selective_scan_jax {

enum ElementType { BF16, F16, F32, F64 };

struct SelectiveScanDescriptor {
  int batch, dim, seqlen, dstate, n_groups, n_chunks, dim_ngroups_ratio;

  bool delta_softplus;
  bool has_z, has_D, has_delta_bias;

  bool is_var_B, is_var_C;
  bool is_complex=false;

  //int A_d_stride, A_dstate_stride;
  //int B_d_stride, B_batch_stride, B_group_stride, B_dstate_stride;
  //int C_d_stride, C_batch_stride, C_group_stride, C_dstate_stride;
  //int u_batch_stride, u_d_stride;
  //int delta_batch_stride, delta_d_stride;
  //int z_batch_stride, z_d_stride;
  //int out_z_batch_stride, out_z_d_stride;
  //int out_batch_stride, out_d_stride;
};

// === Forward functions ===
void gpu_selective_scan_fwd_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

// === Backward functions ===
void gpu_selective_scan_bwd_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace selective_scan_jax