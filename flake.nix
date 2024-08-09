{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { 
            allowUnfree = true;
          };
        };

        python-with-packages = pkgs.python312.withPackages(ps: with ps; [
        #   pipx
        ]);

        cudatoolkit = pkgs.cudaPackages_12_2.cudatoolkit;

        devShellPackages = with pkgs; [
            gcc12
            zlib
            cmake
            gnumake
            python-with-packages
            poetry
            cudatoolkit
            cudaPackages.cuda_cudart
        ];

        development = true;
        venvDir = "./.venv";
      in    
      rec {
        devShell = (pkgs.mkShell.override { stdenv = pkgs.gcc12Stdenv; }) {
          buildInputs = devShellPackages;
          shellHook = if development then (with pkgs; ''
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${gcc12Stdenv.cc.cc.lib}/lib:${cudatoolkit.lib}/lib";
            export PATH="$PATH:${cudatoolkit}/bin";
            export CUDA_PATH="${cudatoolkit}";
            export CUDACXX="${cudatoolkit}/bin/nvcc";
            
            SOURCE_DATE_EPOCH=$(date +%s)
            if [ -d "${venvDir}" ]; then
              echo "Skipping venv creation, '${venvDir}' already exists"
            else
              echo "Creating new venv environment in path: '${venvDir}'"
              ${python-with-packages.python.interpreter} -m venv "${venvDir}"
            fi
            # PYTHONPATH=$PWD/${venvDir}/${python-with-packages.python.sitePackages}/:$PYTHONPATH

            source "${venvDir}/bin/activate"
            # pip install -e .
            # python -c "import jax; print(jax.devices());"
          '') else "";
        };
      }
    );
}