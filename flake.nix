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
                })
                pkgs.poetry2nix.defaultPoetryOverrides
                (pyfinal: pyprev: {
                  torch = pyfinal.torch-bin;
                  contourpy = pyprev.contourpy-bin;
                  pyglet = pkgs.python310.pkgs.pyglet;
                })
              ];
              editablePackageSources = {
                mimicry = "${root}/src";
              };
            };
          in
          poetry-env.env.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ [
              pkgs.pyright
              poetry-env.pkgs.ruff-lsp
              pkgs.ruff
              pkgs.ffmpeg-full
            ];
          });
      });
}
