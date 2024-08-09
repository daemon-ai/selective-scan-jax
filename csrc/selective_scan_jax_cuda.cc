// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace selective_scan_jax;

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  // Forward
  dict["gpu_selective_scan_fwd_f32"] = EncapsulateFunction(gpu_selective_scan_fwd_f32);

  // Backward
  dict["gpu_selective_scan_bwd_f32"] = EncapsulateFunction(gpu_selective_scan_bwd_f32);
  return dict;
}

PYBIND11_MODULE(selective_scan_jax_cuda, m) {
  m.def("registrations", &Registrations);
  m.def("build_selective_scan_descriptor",[](
    uint32_t batch_size, uint32_t dim, uint32_t seqlen, uint32_t dstate, uint32_t n_groups, uint32_t n_chunks, uint32_t dim_ngroups_ratio,
    uint32_t delta_softplus, bool has_z, bool has_D, bool has_delta_bias,
    uint32_t is_var_B, uint32_t is_var_C) {
          return PackDescriptor(SelectiveScanDescriptor{
            batch_size, dim, seqlen, dstate, n_groups, n_chunks, dim_ngroups_ratio,
            delta_softplus>0, has_z, has_D, has_delta_bias,
            is_var_B, is_var_C,
            });
    });
}
}  // namespace