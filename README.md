# Selective Scan JAX (CUDA Accelerated)

Work in Progress

Aims to be a JAX port of [selective_scan for PyTorch](https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan) ([33dc96c84926e58a392861d5ad9d2ee4f4f4a259](https://github.com/state-spaces/mamba/tree/33dc96c84926e58a392861d5ad9d2ee4f4f4a259/csrc/selective_scan))


`nix develop` (optional)  
`poetry install`  (dependencies)
`pip install -e .` (build)

TODO  

- [ ] Python Module
- [ ] Tests
- [ ] Packaging
- [ ] Update and track latest selective_scan

# Some comments
* selective_scan_bwd.cuh
    -> Why don't we use the strides from SSMParamsBase?
    -> dx, dout_z dont we need them?
