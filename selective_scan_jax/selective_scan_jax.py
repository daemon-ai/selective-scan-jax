# -*- coding: utf-8 -*-

__all__ = ["selective_scan"]

from functools import partial

import jax
import jaxlib
import numpy as np
import jax.numpy as jnp
from jax import core, dtypes, lax
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# If the GPU version exists, also register those
try:
    from . import selective_scan_jax_cuda
except ImportError:
    selective_scan_jax_cuda = None
else:
    for _name, _value in selective_scan_jax_cuda.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

#==============================================================================
def selective_scan_fwd(u, delta,
              A, B, C,
              D_, z_, delta_bias_,
              delta_softplus):
    #print("Fwd call", bias)
    has_z = (z_!= None)

    if D_ == None:
        D_ = jnp.nan
    if z_ == None:
        z_ = jnp.nan
    if delta_bias_ == None:
        delta_bias_ = jnp.nan

    if has_z:
        out, x, out_z = _selective_scan_fwd_prim.bind(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus)
        return (out, x, out_z), (u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus)
    else:
        out, x = _selective_scan_fwd_prim.bind(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus)
        return (out, x), (u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus)


def selective_scan_bwd(res, grad):
    #print("Grad(out)", grad)
    #print("Bwd call", bias)
    u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus = res
    if isinstance(z_, jaxlib.xla_extension.ArrayImpl):
        has_z = True
    else:
        has_z = not jnp.isnan(z_)
    print("has_z", has_z)

    if has_z:
        g_out, g_x, g_out_z = grad
    else:
        g_out, g_x = grad
        g_out_z = jnp.nan

    g_u, g_delta, g_A, g_B, g_C, g_D, g_z, g_delta_bias = _selective_scan_bwd_prim.bind(
        g_out, g_x, g_out_z, u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus
    )
    # TODO: gradient for D_, z_, delta_bias_
    return g_u, g_delta, g_A, g_B, g_C, g_D, g_z, g_delta_bias, None

@partial(jax.custom_vjp)
def selective_scan(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus):
    output, _ = selective_scan_fwd(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus)
    return output

# === Differentiation rule ===
selective_scan.defvjp(selective_scan_fwd, selective_scan_bwd)

# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _selective_scan_fwd_abstract(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus):
    inp_dtype = dtypes.canonicalize_dtype(u.dtype)
    w_dtype = dtypes.canonicalize_dtype(A.dtype)
    B_dtype = dtypes.canonicalize_dtype(B.dtype)
    C_dtype = dtypes.canonicalize_dtype(C.dtype)

    assert inp_dtype in [np.float32, np.float16, jnp.bfloat16]
    assert w_dtype in [np.float32, np.csingle]

    is_variable_B = len(B.shape) >= 3
    is_variable_C = len(C.shape) >= 3
    is_complex = w_dtype in [np.csingle]

    # Check Types
    assert dtypes.canonicalize_dtype(delta.dtype) == inp_dtype
    assert B_dtype == (inp_dtype if is_variable_B else w_dtype)
    assert C_dtype == (inp_dtype if is_variable_C else w_dtype)

    # TODO: Check on Cuda device (jax does it implicitly...)

    sizes = u.shape
    batch_size, dim, seqlen, = sizes[0], sizes[1], sizes[2]
    dstate = A.shape[1]
    n_groups = B.shape[1] if is_variable_B else 1

    assert dstate <= 256, "selective_scan only supports state dimension <= 256"
    assert delta.shape[:3] == (batch_size, dim, seqlen)
    assert A.shape[:2] == (dim, dstate)

    if not is_variable_B:
        assert B.shape[:2] == (dim, dstate)
    else:
        assert B.shape[:4] == (batch_size, n_groups, dstate, seqlen if not is_complex else 2*seqlen)
    if not is_variable_C:
        assert C.shape[:2] == (dim, dstate)
    else:
        assert C.shape[:4] == (batch_size, n_groups, dstate, seqlen if not is_complex else 2*seqlen)
    
    #print("D_ shape", D_.shape)
    if D_.shape != ():
        assert dtypes.canonicalize_dtype(D_.dtype) in [np.float32]
        assert D_.shape == (dim, )
    
    if delta_bias_.shape != ():
        assert dtypes.canonicalize_dtype(delta_bias_.dtype) in [np.float32]
        assert delta_bias_.shape == (dim, )

    if z_.shape != ():
        assert dtypes.canonicalize_dtype(z_.dtype) in [np.float32]
        assert z_.shape == (batch_size, dim, seqlen)
        #out_z = np.zeros(z_) 

    n_chunks = (seqlen + 2048 - 1) // 2048
    #out = np.zeros(delta) 
    if z_.shape != ():
        return (ShapedArray(delta.shape, inp_dtype),
                ShapedArray((batch_size, dim, n_chunks, dstate * 2), inp_dtype),
                ShapedArray(z_.shape, np.float32),)
    else:
        return (ShapedArray(delta.shape, inp_dtype),
                ShapedArray((batch_size, dim, n_chunks, dstate * 2), inp_dtype),)

def _selective_scan_bwd_abstract(g_out, g_x, g_out_z, u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus):
    #print()
    assert g_out.shape == delta.shape
    #assert g_x == ..
    #assert g_out_z == ..

    inp_dtype = dtypes.canonicalize_dtype(u.dtype)
    w_dtype = dtypes.canonicalize_dtype(A.dtype)
    _selective_scan_fwd_abstract(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus)
    return (ShapedArray(u.shape, inp_dtype),
            ShapedArray(delta.shape, inp_dtype),
            ShapedArray(A.shape, w_dtype),
            ShapedArray(B.shape, w_dtype),
            ShapedArray(C.shape, w_dtype),
            ShapedArray(D_.shape, w_dtype),
            ShapedArray(z_.shape, w_dtype),
            ShapedArray(delta_bias_.shape, w_dtype),
            )

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

# Helper function
def get_op_precisions(x_np_dtype):
    prec_name = ""
    if x_np_dtype == np.float32:
        prec_name = "_f32"
    elif x_np_dtype == np.float16:
        prec_name = "_f16"
    elif x_np_dtype == jnp.bfloat16:
        prec_name = "_bf16"
    else:
        raise NotImplementedError(f"Unsupported dtype {x_np_dtype}")

    return prec_name

# Lowering:  C++ and/or CUDA interfaces to the JAX XLA backend
def _selective_scan_fwd_lowering(ctx, u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus, *, platform="gpu"):
    # Extract the numpy type of the inputs
    u_aval, A_aval = ctx.avals_in[0], ctx.avals_in[2]
    inp_np_dtype = np.dtype(u_aval.dtype)
    A_np_dtype = np.dtype(A_aval.dtype)

    assert A_np_dtype not in [np.csingle, np.double] # Complex not implemented yet

    u_dtype = mlir.ir.RankedTensorType(u.type)
    u_dims = mlir.ir.RankedTensorType(u.type).shape
    delta_dims = mlir.ir.RankedTensorType(delta.type).shape
    A_dims = mlir.ir.RankedTensorType(A.type).shape
    B_dims = mlir.ir.RankedTensorType(B.type).shape
    C_dims = mlir.ir.RankedTensorType(C.type).shape
    D_dims = mlir.ir.RankedTensorType(D_.type).shape
    z_dims = mlir.ir.RankedTensorType(z_.type).shape
    delta_bias_dims = mlir.ir.RankedTensorType(delta_bias_.type).shape

    all_dims = [mlir.ir.RankedTensorType(v.type).shape for v in 
                [u, delta, A, B, C, D_, z_, delta_bias_]]

    # We dispatch a different call depending on the dtype
    op_name = platform + "_selective_scan_fwd" + get_op_precisions(inp_np_dtype)

    # Output dims
    batch_size, dim, seqlen = u_dims[0:3]
    dstate = A_dims[1]
    n_chunks = (seqlen + 2048 - 1) // 2048
    x_dims = (batch_size, dim, n_chunks, dstate * 2)
    out_dims = [delta_dims, x_dims]
    out_types = [mlir.ir.RankedTensorType.get(delta_dims, u_dtype.element_type),
                 mlir.ir.RankedTensorType.get(x_dims, u_dtype.element_type)]
    if len(z_dims) != 0:
        out_dims += [z_dims]
        out_types += [mlir.ir.RankedTensorType.get(z_dims, u_dtype.element_type)]

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"No CPU implemetnation!")
    elif platform == "gpu":
        if selective_scan_jax_cuda is None:
            raise ValueError(
                "The 'selective_scan_jax_cuda' module was not compiled with CUDA support"
            )

        n_groups = B_dims[1] if len(B_dims)>=3 else 1
        dim_ngroups_ratio = dim // n_groups
        has_z, has_D, has_delta_bias = len(z_dims) > 0, len(D_dims) > 0, len(delta_bias_dims) > 0
        is_var_B, is_var_C = len(B_dims)>=3, len(C_dims)>=3

        print("Dims:", batch_size, dim, seqlen, dstate, n_groups, n_chunks, dim_ngroups_ratio,
            1 if delta_softplus else 0, has_z, has_D, has_delta_bias,
            is_var_B, is_var_C)
        opaque = selective_scan_jax_cuda.build_selective_scan_descriptor(
            batch_size, dim, seqlen, dstate, n_groups, n_chunks, dim_ngroups_ratio,
            1 if delta_softplus else 0, has_z, has_D, has_delta_bias,
            is_var_B, is_var_C)

        return custom_call(
            op_name,
            # Output types
            result_types=out_types,
            result_layouts=default_layouts(*out_dims),
            # The inputs:
            operands=[u, delta, A, B, C, D_, z_, delta_bias_],
            operand_layouts=default_layouts(*all_dims),
            backend_config=opaque
        ).results

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )

def _selective_scan_bwd_lowering(ctx, g_out, g_x, g_out_z, u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus, *, platform="gpu"):
    #raise NotImplementedError("Bwd not implemented")

    # === Extract dtypes ===
    u_aval = ctx.avals_in[1]
    inp_np_dtype = np.dtype(u_aval.dtype)

    u_dtype = mlir.ir.RankedTensorType(u.type)
    u_dims = mlir.ir.RankedTensorType(u.type).shape
    delta_dims = mlir.ir.RankedTensorType(delta.type).shape
    w_dtype = mlir.ir.RankedTensorType(A.type)
    A_dims = mlir.ir.RankedTensorType(A.type).shape
    B_dims = mlir.ir.RankedTensorType(B.type).shape
    C_dims = mlir.ir.RankedTensorType(C.type).shape
    D_dims = mlir.ir.RankedTensorType(D_.type).shape
    z_dims = mlir.ir.RankedTensorType(z_.type).shape
    delta_bias_dims = mlir.ir.RankedTensorType(delta_bias_.type).shape

    all_dims = [mlir.ir.RankedTensorType(v.type).shape for v in
                [g_out, g_x, g_out_z, u, delta, A, B, C, D_, z_, delta_bias_]]

    # === Dispatch a different call depending on the dtype ===
    op_name = platform + "_selective_scan_bwd" + get_op_precisions(inp_np_dtype)

    # === Output dims ===
    batch_size, dim, seqlen = u_dims[0:3]
    dstate = A_dims[1]
    n_chunks = (seqlen + 2048 - 1) // 2048
    x_dims = (batch_size, dim, n_chunks, dstate * 2)

    grad_out_dims = [u_dims, delta_dims, A_dims, B_dims, C_dims, D_dims, z_dims, delta_bias_dims]
    grad_out_types = [mlir.ir.RankedTensorType.get(u_dims, u_dtype.element_type),
                      mlir.ir.RankedTensorType.get(delta_dims, u_dtype.element_type),
                      mlir.ir.RankedTensorType.get(A_dims, w_dtype.element_type),
                      mlir.ir.RankedTensorType.get(B_dims, w_dtype.element_type),
                      mlir.ir.RankedTensorType.get(C_dims, w_dtype.element_type),
                      mlir.ir.RankedTensorType.get(D_dims, w_dtype.element_type),
                      mlir.ir.RankedTensorType.get(z_dims, w_dtype.element_type),
                      mlir.ir.RankedTensorType.get(delta_bias_dims, w_dtype.element_type),
                      ]
    print("=== Dims ===", len(grad_out_dims), len(grad_out_types))
    # TODO grad for D_, z_, delta_softplus_
    #if len(D_dims) != 0:
    #    out_dims += [D_dims]
    #    out_types += [mlir.ir.RankedTensorType.get(D_dims, u_dtype.element_type)]

    # === Switch on platform ===
    if platform == "cpu":
        raise NotImplementedError(f"No CPU implementation!")
    elif platform == "gpu":
        if selective_scan_jax_cuda is None:
            raise ValueError(
                "The 'selective_scan_jax_cuda' module was not compiled with CUDA support"
            )
        # === Descriptor ===
        n_groups = B_dims[1] if len(B_dims)>=3 else 1
        dim_ngroups_ratio = dim // n_groups
        has_z, has_D, has_delta_bias = len(z_dims) > 0, len(D_dims) > 0, len(delta_bias_dims) > 0
        is_var_B, is_var_C = len(B_dims)>=3, len(C_dims)>=3

        print("Dims:", batch_size, dim, seqlen, dstate, n_groups, n_chunks, dim_ngroups_ratio,
            1 if delta_softplus else 0, has_z, has_D, has_delta_bias,
            is_var_B, is_var_C)

        opaque = selective_scan_jax_cuda.build_selective_scan_descriptor(
            batch_size, dim, seqlen, dstate, n_groups, n_chunks, dim_ngroups_ratio,
            1 if delta_softplus else 0, has_z, has_D, has_delta_bias,
            is_var_B, is_var_C)

        # === Call operation ===
        return custom_call(
            op_name,
            # Output types
            result_types=grad_out_types,
            result_layouts=default_layouts(*grad_out_dims),
            # The inputs:
            operands=[g_out, g_x, g_out_z, u, delta, A, B, C, D_, z_, delta_bias_],
            operand_layouts=default_layouts(*all_dims),
            backend_config=opaque
        ).results

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )
# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_selective_scan_fwd_prim = core.Primitive("selective_scan_fwd")
_selective_scan_fwd_prim.multiple_results = True
_selective_scan_fwd_prim.def_impl(partial(xla.apply_primitive, _selective_scan_fwd_prim))
_selective_scan_fwd_prim.def_abstract_eval(_selective_scan_fwd_abstract)

_selective_scan_bwd_prim = core.Primitive("selective_scan_bwd")
_selective_scan_bwd_prim.multiple_results = True
_selective_scan_bwd_prim.def_impl(partial(xla.apply_primitive, _selective_scan_bwd_prim))
_selective_scan_bwd_prim.def_abstract_eval(_selective_scan_bwd_abstract)


# Connect the XLA translation rules for JIT compilation
for platform in ["gpu"]:
    mlir.register_lowering(
        _selective_scan_fwd_prim,
        partial(_selective_scan_fwd_lowering, platform=platform),
        platform=platform)

    mlir.register_lowering(
        _selective_scan_bwd_prim,
        partial(_selective_scan_bwd_lowering, platform=platform),
        platform=platform)