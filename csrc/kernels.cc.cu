// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "kernels.h"
#include "kernel_helpers.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

//#include "cuda_src/selective_scan.h"
#include "cuda_src/cuda_common.h"
#include "cuda_src/selective_scan.h"
#include "cuda_src/selective_scan_common.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "cuda_src/selective_scan_fwd_kernel.cuh"
#include "cuda_src/selective_scan_bwd_kernel.cuh"

namespace selective_scan_jax {

namespace {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

// === DEBUG ===
template <typename T>
__global__ void set_val(int size, T val, T* out){
  out[size] = val;
}

void set_params(const SelectiveScanDescriptor &d, SSMParamsBase &params){
  params.batch = d.batch;
  params.dim = d.dim;
  params.seqlen = d.seqlen;
  params.dstate = d.dstate;
  params.n_groups = d.n_groups;
  params.n_chunks = d.n_chunks;
  params.dim_ngroups_ratio = d.dim_ngroups_ratio;

  params.is_variable_B = d.is_var_B;
  params.is_variable_C = d.is_var_C;

  params.delta_softplus = d.delta_softplus;

  params.A_d_stride = d.dstate;
  params.A_dstate_stride = 1;

  if(!d.is_var_B){
    params.B_d_stride = d.dstate;
    params.B_dstate_stride = 1;
  }
  else{
    params.B_batch_stride = d.dstate*d.seqlen*(d.is_complex?2:1);//TODO: *=chunk_size?
    params.B_group_stride = d.dstate*d.seqlen*(d.is_complex?2:1);
    params.B_dstate_stride = d.seqlen*(d.is_complex?2:1);
  }

  if(!d.is_var_C){
    params.C_d_stride = d.dstate;
    params.C_dstate_stride = 1;
  }
  else{
    params.C_batch_stride = d.dstate*d.seqlen*(d.is_complex?2:1);//TODO: *=chunk_size?
    params.C_group_stride = d.dstate*d.seqlen*(d.is_complex?2:1);
    params.C_dstate_stride = d.seqlen*(d.is_complex?2:1);
  }

  params.u_batch_stride = d.dim*d.seqlen;
  params.u_d_stride = d.seqlen;
  params.delta_batch_stride = d.dim*d.seqlen;
  params.delta_d_stride = d.seqlen;
  params.z_batch_stride = d.dim*d.seqlen;
  params.z_d_stride = d.seqlen;
  params.out_batch_stride = d.dim*d.seqlen;
  params.out_d_stride = d.seqlen;
  params.out_z_batch_stride = d.dim*d.seqlen;
  params.out_z_d_stride = d.seqlen;
}

void set_params_bwd(SSMParamsBwd &params){
  params.dout_batch_stride = params.out_batch_stride;
  params.dout_d_stride = params.out_d_stride;
  params.dA_d_stride = params.A_d_stride;
  params.dA_dstate_stride = params.A_dstate_stride;
  params.dB_batch_stride = params.B_batch_stride;
  params.dB_group_stride = params.B_group_stride;
  params.dB_d_stride = params.B_d_stride;
  params.dB_dstate_stride = params.B_dstate_stride;
  params.dC_batch_stride = params.C_batch_stride;
  params.dC_group_stride = params.C_group_stride;
  params.dC_d_stride = params.C_d_stride;
  params.dC_dstate_stride = params.C_dstate_stride;
  params.du_batch_stride = params.u_batch_stride;
  params.du_d_stride = params.u_d_stride;
  params.dz_batch_stride = params.z_batch_stride;
  params.dz_d_stride = params.z_d_stride;
  params.ddelta_batch_stride = params.delta_batch_stride;
  params.ddelta_d_stride = params.delta_d_stride;
}

template <typename input_t, typename weight_t>
inline void apply_selective_scan_fwd(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
  // TODO: const?
  const SelectiveScanDescriptor &d = *UnpackDescriptor<SelectiveScanDescriptor>(opaque, opaque_len);

  // TODO: const for the first 3
  input_t *u = reinterpret_cast<input_t *>(buffers[0]);
  input_t *delta = reinterpret_cast<input_t *>(buffers[1]);
  weight_t *A = reinterpret_cast<weight_t *>(buffers[2]);
  weight_t *B = reinterpret_cast<weight_t *>(buffers[3]);
  weight_t *C = reinterpret_cast<weight_t *>(buffers[4]);
  weight_t *D_ = reinterpret_cast<weight_t *>(buffers[5]);
  weight_t *z_ = reinterpret_cast<weight_t *>(buffers[6]);
  weight_t *delta_bias_ = reinterpret_cast<weight_t *>(buffers[7]);

  input_t *out = reinterpret_cast<weight_t *>(buffers[8]);
  input_t *x = reinterpret_cast<input_t *>(buffers[9]);
  input_t *out_z = nullptr;
  if(d.has_z){
    out_z = reinterpret_cast<input_t *>(buffers[10]);
  }

  SSMParamsBase params;
  set_params(d, params);

  params.u_ptr = u;
  params.delta_ptr = delta;
  params.A_ptr = A;
  params.B_ptr = B;
  params.C_ptr = C;
  params.D_ptr = d.has_D?D_:nullptr;
  params.z_ptr = d.has_z?z_:nullptr;
  params.delta_bias_ptr = d.has_delta_bias?delta_bias_:nullptr;

  params.out_ptr = out;
  params.x_ptr = x;
  params.out_z_ptr = out_z;

  //[DEBUG]
  //debug_params<T>(params, out);
  //set_val<<<1,1>>>(0, (input_t)d.batch, out);
  //set_val<<<1,1>>>(1, (input_t)d.dim, out);
  //set_val<<<1,1>>>(2, (input_t)d.seqlen, out);
  //set_val<<<1,1>>>(3, (input_t)d.dstate, out);
  //set_val<<<1,1>>>(4, (input_t)d.n_groups, out);
  //set_val<<<1,1>>>(5, (input_t)d.n_chunks, out);
  //set_val<<<1,1>>>(6, (input_t)d.dim_ngroups_ratio, out);
  //set_val<<<1,1>>>(7, (input_t)d.is_var_B, out);
  //set_val<<<1,1>>>(8, (input_t)d.is_var_C, out);
  //set_val<<<1,1>>>(9, (input_t)d.delta_softplus, out);
  //set_val<<<1,1>>>(10, (input_t)(D_==nullptr), out);
  //set_val<<<1,1>>>(11, (input_t)(z_==nullptr), out);
  //set_val<<<1,1>>>(0, (input_t)d.has_z, out);
  //set_val<<<1,1>>>(1, (input_t)d.has_D, out);
  //set_val<<<1,1>>>(2, (input_t)d.has_delta_bias, out);
  //return;

  selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ThrowIfError(cudaGetLastError());
}

template <typename input_t, typename weight_t>
inline void apply_selective_scan_bwd(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
    const SelectiveScanDescriptor &d = *UnpackDescriptor<SelectiveScanDescriptor>(opaque, opaque_len);

    // TODO: const for the first 3
    input_t *g_out = reinterpret_cast<input_t *>(buffers[0]);
    input_t *g_x = reinterpret_cast<input_t *>(buffers[1]);
    input_t *g_out_z = reinterpret_cast<input_t *>(buffers[2]);

    input_t *u = reinterpret_cast<input_t *>(buffers[3]);
    input_t *delta = reinterpret_cast<input_t *>(buffers[4]);
    weight_t *A = reinterpret_cast<weight_t *>(buffers[5]);
    weight_t *B = reinterpret_cast<weight_t *>(buffers[6]);
    weight_t *C = reinterpret_cast<weight_t *>(buffers[7]);
    weight_t *D_ = reinterpret_cast<weight_t *>(buffers[8]);
    weight_t *z_ = reinterpret_cast<weight_t *>(buffers[9]);
    weight_t *delta_bias_ = reinterpret_cast<weight_t *>(buffers[10]);

    input_t *g_u = reinterpret_cast<input_t *>(buffers[11]);
    input_t *g_delta = reinterpret_cast<input_t *>(buffers[12]);
    weight_t *g_A = reinterpret_cast<weight_t *>(buffers[13]);
    weight_t *g_B = reinterpret_cast<weight_t *>(buffers[14]);
    weight_t *g_C = reinterpret_cast<weight_t *>(buffers[15]);
    weight_t *g_D = reinterpret_cast<weight_t *>(buffers[16]);
    weight_t *g_z = reinterpret_cast<weight_t *>(buffers[17]);
    weight_t *g_delta_bias = reinterpret_cast<weight_t *>(buffers[18]);

    SSMParamsBwd params;
    set_params(d, params);
    set_params_bwd(params);

    params.u_ptr = u;
    params.delta_ptr = delta;
    params.A_ptr = A;
    params.B_ptr = B;
    params.C_ptr = C;
    params.D_ptr = d.has_D?D_:nullptr;
    params.z_ptr = d.has_z?z_:nullptr;
    params.delta_bias_ptr = d.has_delta_bias?delta_bias_:nullptr;

    params.dout_ptr = g_out;
    params.du_ptr = g_u;
    params.ddelta_ptr = g_delta;
    params.dA_ptr = g_A;
    params.dB_ptr = g_B;
    params.dC_ptr = g_C;
    params.dD_ptr = d.has_D?g_D:nullptr;
    params.dz_ptr = d.has_z?g_z:nullptr;
    params.ddelta_bias_ptr = d.has_delta_bias?g_delta_bias:nullptr;

    selective_scan_bwd_cuda<input_t, weight_t>(params, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ThrowIfError(cudaGetLastError());
}

}  // namespace

// === Forward functions ===
void gpu_selective_scan_fwd_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_selective_scan_fwd<float, float>(stream, buffers, opaque, opaque_len);
}

// === Backward functions ===
void gpu_selective_scan_bwd_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_selective_scan_bwd<float, float>(stream, buffers, opaque, opaque_len);
}

}  // namespace selective_scan_jax