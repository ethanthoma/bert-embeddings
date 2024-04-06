{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          system = "x86_64-linux";
          config.allowUnfree = true;
          cudaSupport = true;
        };
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication overrides;
      in
      {
        packages = {
          myapp = mkPoetryApplication {
            projectDir = self;
            python = pkgs.python311;
            overrides = overrides.withDefaults
              (self: super: {
                safetensors = super.safetensors.overridePythonAttrs (
                  old: {
                    cargoDeps = pkgs.rustPlatform.fetchCargoTarball {
                      inherit (old) src;
                      name = "${old.pname}-${old.version}";
                      sourceRoot = "${old.pname}-${old.version}/bindings/python";
                      sha256 = "sha256-l+iCrRDEpTisrPh39cs3NiP2fAuLZOFp3V8fyrDJLX0=";
                    };
                    cargoRoot = "bindings/python";
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools super.setuptools-rust ];
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                      pkgs.rustc
                      pkgs.cargo
                      pkgs.rustPlatform.cargoSetupHook
                      self.maturin
                    ];
                  }
                );

                tokenizers = super.tokenizers.overridePythonAttrs (
                  old: {
                    cargoDeps = pkgs.rustPlatform.fetchCargoTarball {
                      inherit (old) src;
                      name = "${old.pname}-${old.version}";
                      sourceRoot = "${old.pname}-${old.version}/bindings/python";
                      sha256 = "sha256-iisLwEWuokGyZ+8NSxzPwXkssNkPphlXpHDMEvU7nmE=";
                    };
                    cargoRoot = "bindings/python";
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools super.setuptools-rust ];
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                      pkgs.rustc
                      pkgs.cargo
                      pkgs.rustPlatform.cargoSetupHook
                      self.maturin
                    ];
                  }
                );

                maturin = super.maturin.overridePythonAttrs (
                  old: {
                    cargoDeps = pkgs.rustPlatform.fetchCargoTarball {
                      inherit (old) src;
                      name = "${old.pname}-${old.version}";
                      hash = "sha256-hPyPMQm/Oege0PPjYIrd1fEDOGqoQ1ffS2l6o8je4t4=";
                    };
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools super.setuptools-rust ];
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.cargo pkgs.rustc pkgs.rustPlatform.cargoSetupHook ];
                  }
                );

                nvidia-cusparse-cu12 = super.nvidia-cusparse-cu12.overridePythonAttrs (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.cudaPackages.libnvjitlink ];
                  }
                );

                nvidia-cusolver-cu12 = super.nvidia-cusolver-cu12.overridePythonAttrs (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.cudaPackages.libnvjitlink pkgs.cudaPackages.libcublas pkgs.cudaPackages.libcusparse ];
                  }
                );

                textblob = super.textblob.overridePythonAttrs (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.flit-core ];
                  }
                );

                pandas = super.pandas.overridePythonAttrs (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools pkgs.python311Packages.cython_3 ];
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.ninja pkgs.pkg-config pkgs.python311Packages.cython_3];
                  }
                );
              });
          };
          default = self.packages.${system}.myapp;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.myapp ];
          packages = [ pkgs.poetry ];
        };
      });
}
