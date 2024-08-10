import pytest
import jax
import numpy as np
from jax import numpy as jnp
import jax._src.test_util as jtu

from jax.test_util import check_grads
from selective_scan_jax import selective_scan
from utils import selective_scan_ref, selective_scan_inner_ref

import torch
import pytest
from einops import rearrange

jax.config.update("jax_enable_x64", True)


def test_selective_scan_fwd(is_var = False):
    batch_size = 2
    dim = 4
    dstate = 8
    seqlen = 128
    A = (-0.5 * np.random.rand(dim, dstate)).astype(np.float32)
    if is_var:
        B = np.random.normal(size = (batch_size, 1, dstate, seqlen)).astype(np.float32)
        C = np.random.normal(size = (batch_size, 1, dstate, seqlen)).astype(np.float32)
    else:
        B = np.random.normal(size = (dim, dstate)).astype(np.float32)
        C = np.random.normal(size = (dim, dstate)).astype(np.float32)

    u = np.random.normal(size= (batch_size, dim, seqlen)).astype(np.float32)
    delta = 0.5 * np.random.rand(batch_size, dim, seqlen).astype(np.float32)

    z_ = np.random.rand(batch_size, dim, seqlen).astype(np.float32)
    D_ = np.random.rand(dim).astype(np.float32)
    delta_bias_ = 0.5*np.random.rand(dim).astype(np.float32)    

    if is_var:
        out, x, out_z = jax.jit(selective_scan)(
            u, delta, A, B, C, D_=D_, z_=z_,
            delta_bias_=delta_bias_, delta_softplus=True
        )
    else:
        out_z = np.zeros(1)
        out, x = jax.jit(selective_scan)(
            u, delta, A, B, C, D_=None, z_=None,
            delta_bias_=None, delta_softplus=True
        )

    print(out, x, out.shape, x.shape, out_z.shape)

def test_selective_scan_bwd():
    def loss(*args, **kwargs):
        out, x = selective_scan(*args, **kwargs)
        return -jnp.mean(out**2)

    batch_size = 2
    dim = 4
    dstate = 8
    seqlen = 128
    A = (-0.5 * np.random.rand(dim, dstate)).astype(np.float32)
    B = np.random.normal(size = (dim, dstate)).astype(np.float32)
    C = np.random.normal(size = (dim, dstate)).astype(np.float32)

    u = np.random.normal(size= (batch_size, dim, seqlen)).astype(np.float32)
    delta = 0.5 * np.random.rand(batch_size, dim, seqlen).astype(np.float32)

    z_ = np.random.rand(batch_size, dim, seqlen).astype(np.float32)
    D_ = np.random.rand(dim).astype(np.float32)
    delta_bias_ = 0.5*np.random.rand(dim).astype(np.float32)

    #check_grads(loss, (x, w, b), modes=["rev"], order=1) # Error for float16
    
    out = jax.grad(loss, argnums=0)(u, delta, A, B, C, D_=None, z_=None, delta_bias_=None, delta_softplus=True)
    print("Gradient", out, out.shape)

# ===============================
#          UNIT TEST
# ===============================
from causal_conv1d_jax import causal_conv1d

def selective_scan_fn(xz, conv1d_weight, conv1d_bias, x_proj_weight,
                              delta_proj_weight, out_proj_weight, out_proj_bias,
                              A, B, C, D,
                              delta_bias, delta_softplus,
                              debug = False):
    is_complex = jnp.iscomplexobj(A)
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not is_complex else 2)
    conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
    x, z = jnp.split(xz, 2, axis=1)
    conv1d_out = jax.jit(causal_conv1d)(x, conv1d_weight, conv1d_bias, activation=True)

    x_dbl = rearrange(conv1d_out, 'b d l -> (b l) d') @ x_proj_weight.T  # (bl d)
    delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].T, "d (b l) -> b d l", l = L)

    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
        if not is_complex:
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L)
        else:
            B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2)

    if C is None:  # variable C
        C = x_dbl[:, -d_state:]  # (bl dstate)
        if not is_complex:
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L)
        else:
            C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2)

    if debug:
        print('U', np.shape(conv1d_out))
        print('delta', np.shape(delta))
        print('A', np.shape(A))
        print('B', np.shape(B))
        print('C', np.shape(C))
        print('D', np.shape(D))
        print('z', np.shape(z))
        print('delta_bias', np.shape(delta_bias))
    out, scan_intermediates, out_z = selective_scan(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
    )

    print("CUDA out sizes:", out.shape, scan_intermediates.shape, out_z.shape)
    if debug:
        print("Out_z shape: ", out_z.shape)

    if out_proj_bias:
        return (rearrange(out_z, "b d l -> b l d") @ out_proj_weight.T) + out_proj_bias
    else:
        return (rearrange(out_z, "b d l -> b l d") @ out_proj_weight.T)


@pytest.mark.parametrize('wtype', [np.float32, np.complex64])
@pytest.mark.parametrize('itype', [np.float32])
@pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize("is_variable_C", [False, True])
@pytest.mark.parametrize("is_variable_B", [False, True])
def unit_test_selective_scan(TEST_NAME, is_variable_B, is_variable_C, seqlen, itype, wtype):
    # set seed
    key = jax.random.PRNGKey(0)
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)

    # === INIT INPUT VARIABLES ===

    batch_size = 2
    dim = 768
    dstate = 8
    dt_rank = 48
    is_complex = (wtype == np.complex64) or (wtype == np.cdouble)

    key, subkey = jax.random.split(key)
    xz = jax.random.normal(subkey,
                          shape=(batch_size, 2 * dim, seqlen),
                          dtype=itype)
    key, subkey = jax.random.split(key)
    conv1d_weight = jax.random.normal(subkey,
                          shape=(dim, 1, 3),
                          dtype=np.float32)
    key, subkey = jax.random.split(key)
    conv1d_bias = jax.random.normal(subkey,
                          shape=(dim,),
                          dtype=np.float32)
    key, subkey = jax.random.split(key)
    x_proj_weight = jax.random.normal(subkey,
                          shape=(dt_rank + (bool(is_variable_B) + bool(is_variable_C)) * dstate * (1 if not is_complex else 2), dim),
                          dtype=itype)
    key, subkey = jax.random.split(key)
    delta_proj_weight = jax.random.normal(subkey,
                          shape=(dim, dt_rank),
                          dtype=itype)
    key, subkey = jax.random.split(key)
    out_proj_weight = jax.random.normal(subkey,
                          shape=(dim // 2 if is_complex else dim, dim),
                          dtype=itype)
    out_proj_bias = None

    key, subkey = jax.random.split(key)
    A = -0.5 * jax.random.uniform(subkey,
                          shape=(dim, dstate),
                          dtype=wtype)
    key, subkey = jax.random.split(key)
    B = jax.random.normal(subkey,
                          shape=(dim, dstate),
                          dtype=wtype) \
        if not is_variable_B else None
    key, subkey = jax.random.split(key)
    C = jax.random.normal(subkey,
                          shape=(dim, dstate),
                          dtype=wtype) \
        if not is_variable_C else None
    key, subkey = jax.random.split(key)
    D = jax.random.normal(subkey, shape=(dim,), dtype=np.float32)
    key, subkey = jax.random.split(key)
    delta_bias = 0.5 * jax.random.normal(subkey, shape=(dim,), dtype=np.float32)
    B_proj_bias = None
    C_proj_bias = None

    # === Inference ref ===
    xz_ref = torch.tensor(np.array(xz), requires_grad=True).cuda()
    conv1d_weight_ref = torch.tensor(np.array(conv1d_weight), requires_grad=True).cuda()
    conv1d_bias_ref = torch.tensor(np.array(conv1d_bias), requires_grad=True).cuda()
    x_proj_weight_ref = torch.tensor(np.array(x_proj_weight), requires_grad=True).cuda()
    delta_proj_weight_ref = torch.tensor(np.array(delta_proj_weight), requires_grad=True).cuda()
    out_proj_weight_ref = torch.tensor(np.array(out_proj_weight), requires_grad=True).cuda()
    out_proj_bias_ref = torch.tensor(np.array(out_proj_bias), requires_grad=True).cuda() if out_proj_bias is not None else None

    A_ref = torch.tensor(np.array(A), requires_grad=True).cuda()
    B_ref = torch.tensor(np.array(B), requires_grad=True).cuda() if B is not None else None
    C_ref = torch.tensor(np.array(C), requires_grad=True).cuda() if C is not None else None
    D_ref = torch.tensor(np.array(D), requires_grad=True).cuda()

    delta_bias_ref = torch.tensor(np.array(delta_bias), requires_grad=True).cuda() if delta_bias is not None else None

    out_ref = selective_scan_inner_ref(xz_ref, conv1d_weight_ref, conv1d_bias_ref, x_proj_weight_ref,
                              delta_proj_weight_ref, out_proj_weight_ref, out_proj_bias_ref,
                              A_ref, B_ref, C_ref, D_ref,
                              delta_bias=delta_bias_ref, delta_softplus=True)

    print("Out shape:", out_ref.shape)
    #print("Out tensor:", out_ref)

    out = jax.jit(selective_scan_fn)(xz, conv1d_weight, conv1d_bias, x_proj_weight,
                   delta_proj_weight, out_proj_weight, out_proj_bias,
                   A, B, C, D,
                   delta_bias, delta_softplus=True
    )
    #print(out, out.shape)

    max_diff = jnp.abs(out - out_ref.detach().cpu().numpy()).max()
    mean_diff = jnp.abs(out - out_ref.detach().cpu().numpy()).mean()
    mean_abs_val = np.mean(np.abs(out))
    print(f"{TEST_NAME} Output max diff: {max_diff:.3}    [Mean (abs) output: {mean_abs_val}]")
    print(f"{TEST_NAME} Output mean diff: {mean_diff:.3f} [Relative err: {mean_diff/mean_abs_val}]")

    """
    # === TEST BWD CONSISTENCY WITH REFERENCE ===
    def l2_loss(xz, conv1d_weight, conv1d_bias, x_proj_weight,
                   delta_proj_weight, out_proj_weight, out_proj_bias,
                   A, B, C, D,
                   delta_bias, delta_softplus):
        out = selective_scan_fn(xz, conv1d_weight, conv1d_bias, x_proj_weight,
                    delta_proj_weight, out_proj_weight, out_proj_bias,
                    A, B, C, D,
                    delta_bias, delta_softplus
        )
        return -jnp.mean(out**2)
    g_xz, g_conv1d_weight, g_A = jax.grad(l2_loss, argnums=(0,1,7))(xz, conv1d_weight, conv1d_bias, x_proj_weight,
                   delta_proj_weight, out_proj_weight, out_proj_bias,
                   A, B, C, D,
                   delta_bias, True)

    mse = (-(out_ref-0)**2).mean()
    xz_ref.retain_grad()
    conv1d_weight_ref.retain_grad()
    A_ref.retain_grad()
    mse.backward()

    print(f"{TEST_NAME} dxz max diff: {jnp.abs(g_xz - xz_ref.grad.detach().cpu().numpy()).max()}")
    print(f"{TEST_NAME} dconv1d_weight max diff: {jnp.abs(g_conv1d_weight - conv1d_weight_ref.grad.detach().cpu().numpy()).max()}")
    print(f"{TEST_NAME} dA max diff: {jnp.abs(g_A - A_ref.grad.detach().cpu().numpy()).max()}")

    return
    """

if __name__ == "__main__":
    #test_selective_scan_fwd(is_var=False)
    #test_selective_scan_fwd(is_var=True)
    #test_selective_scan_bwd()

    #unit_test_selective_scan("TEMP", False, False, 128, np.float32, np.complex64)
    #unit_test_selective_scan("TEMP", True, True, 128, np.float32, np.complex64)

    #unit_test_selective_scan("TEMP", True, True, 128, np.float32, np.float32)
    unit_test_selective_scan("TEMP", False, False, 128, np.float32, np.float32)