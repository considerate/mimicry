{
  description = "Julia2Nix development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    # flake-utils.inputs.nixpkgs.follows = "nixpkgs";
    nixgl.url = "github:guibou/nixGL";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  };

  outputs =
    inputs @ { self
    , ...
    }:
    (
      inputs.flake-utils.lib.eachDefaultSystem
        (system:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              permittedInsecurePackages = [
                "qtwebkit-5.212.0-alpha4"
              ];
              allowUnfree = true;
            };
            time = 0;
            overlays = [
              inputs.nixgl.overlay
              self.overlays.default
            ];
          };
          targets = pkgs: [
            pkgs.julia
            # pkgs.qt5.full
            pkgs.glib.dev
            pkgs.xorg.libX11
            pkgs.xorg.libXcursor
            pkgs.xorg.libXrandr
            pkgs.xorg.libXt
            pkgs.xorg.libXrender
            pkgs.xorg.libXext
            pkgs.xorg.libXinerama
            pkgs.glfw
            # pkgs.xvfb
            pkgs.mesa
            # pkgs.cairo.dev
            pkgs.dbus.dev
            pkgs.iana-etc
            pkgs.dbus.lib
            # from davinci
            pkgs.udev
            pkgs.librsvg
            pkgs.libGLU
            pkgs.libGL
            # pkgs.xorg.libICE
            # pkgs.xorg.libSM
            # pkgs.xorg.libXxf86vm
            pkgs.alsa-lib
            pkgs.expat
            pkgs.zlib
            pkgs.libuuid
            pkgs.bzip2
            pkgs.libtool
            pkgs.ocl-icd
            # dev environment
            pkgs.neovim
            pkgs.glxinfo

            pkgs.silver-searcher
            # memory
            # pkgs.valgrind
            # pkgs.zee
            # pkgs.helix
          ];
          fhs = pkgs.buildFHSUserEnv {
            name = "mimicry-env";
            targetPkgs = targets;
            multiPkgs = pkgs: [
              pkgs.udev
              pkgs.alsa-lib
            ];
            runScript = ''bash --noprofile --norc'';
          };
        in
        {
          legacyPackages = pkgs;
          packages.fhs = fhs;
          devShells.fhs = fhs.env.overrideAttrs (old: {
            # LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:${pkgs.lib.strings.makeLibraryPath (targets pkgs)}";
            #shellHook = (old.shellHook or "") + ''
            #  export LD_LIBRARY_PATH="$(nixGLNvidia printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH"
            #'';
          });
          devShells.default = pkgs.stdenv.mkDerivation {
            name = "julia-shell";
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:${pkgs.lib.strings.makeLibraryPath (targets pkgs)}";
            buildInputs = [
              pkgs.julia
            ];
          };
        })
    )
    // {
      overlays.default = final: prev: { };
    };
}
