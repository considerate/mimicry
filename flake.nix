{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixgl.url = "github:guibou/nixGL";
    nixpkgs.url = "github:NixOS/nixpkgs";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
          overlays = [
            inputs.nixgl.overlay
          ];
        };
      in
      {
        legacyPackages = pkgs;
        devShells.default =
          let
            pwd = builtins.getEnv "PWD";
            root = if pwd != "" then pwd else ./.;
            poetry-env = pkgs.poetry2nix.mkPoetryEnv {
              projectDir = ./.;
              overrides = [
                (pyfinal: pyprev: {
                  contourpy-bin = pyprev.contourpy.override {
                    preferWheel = true;
                  };
                  pandas-bin = pyprev.pandas.override {
                    preferWheel = true;
                  };
                  pygame-bin = pyprev.pygame.override {
                    preferWheel = true;
                  };
                  pyqt6 = pkgs.python310.pkgs.pyqt6;
                  farama-notifications = pyprev.farama-notifications.overridePythonAttrs (old: {
                    nativeBuildInputs = old.nativeBuildInputs ++ [
                      pyfinal.setuptools
                    ];
                  });
                  gymnasium = pyprev.gymnasium.overridePythonAttrs (old: {
                    nativeBuildInputs = old.nativeBuildInputs ++ [
                      pyfinal.setuptools
                    ];
                  });
                  autorom-accept-rom-license = pyprev.autorom-accept-rom-license.overridePythonAttrs (old: {
                    nativeBuildInputs = old.nativeBuildInputs ++ [
                      pyfinal.setuptools
                    ];
                  });
                  werkzeug = pyprev.werkzeug.overridePythonAttrs (old: {
                    nativeBuildInputs = old.nativeBuildInputs ++ [
                      pyfinal.flit-core
                    ];
                  });
                })
                pkgs.poetry2nix.defaultPoetryOverrides
                (pyfinal: pyprev: {
                  torch = pyfinal.torch-bin;
                  contourpy = pyprev.contourpy-bin;
                  pandas = pyprev.pandas-bin;
                  pygame = pyprev.pygame-bin;
                  pyglet = pkgs.python310.pkgs.pyglet;
                  pyqt6 = null;
                  matplotlib = pyfinal.callPackage "${pkgs.path}/pkgs/development/python-modules/matplotlib" {
                    Cocoa = null;
                  };
                })
              ];
              preferWheels = true;
              editablePackageSources = {
                mimicry = "${root}/src";
              };
            };
          in
          poetry-env.env.overrideAttrs (old: {
            shellHook = ''
              export LD_LIBRARY_PATH=/run/opengl-driver/lib
            '';
            nativeBuildInputs = old.nativeBuildInputs ++ [
              pkgs.pyright
              poetry-env.pkgs.ruff-lsp
              poetry-env.pkgs.yappi
              pkgs.ruff
              pkgs.ffmpeg-full
            ];
          });
      });
}
