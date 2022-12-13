{
  description = "Julia2Nix development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.nixpkgs.follows = "nixpkgs";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.05";
  };

  outputs =
    inputs @ { self
    , ...
    }:
    (
      inputs.flake-utils.lib.eachDefaultSystem
        (system:
        let
          pkgs = inputs.nixpkgs.legacyPackages.${system}.appendOverlays [
            self.overlays.default
          ];
          fhs = pkgs.buildFHSUserEnv {
            name = "mimicry-env";
            targetPkgs = pkgs: [
              pkgs.julia_17-bin
              pkgs.qt5.qtbase
              pkgs.glib.dev
              pkgs.xorg.libX11
              pkgs.xorg.libXcursor
              pkgs.xorg.libXrandr
              pkgs.xorg.libXt
              pkgs.xorg.libXrender
              pkgs.xorg.libXext
              pkgs.mesa
              pkgs.cairo.dev
              pkgs.dbus.dev
              pkgs.iana-etc
              pkgs.dbus.lib
              # from davinci
              pkgs.udev
              pkgs.librsvg
              # pkgs.libGLU
              # pkgs.libGL
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
              # memory
              pkgs.valgrind
            ];
            multiPkgs = pkgs: (with pkgs;
              [
                udev
                alsa-lib
              ]);
            runScript = "bash";
            profile = ''
            '';
          };
        in
        {
          packages.default = fhs;
          devShells.default = fhs.env;
        })
    )
    // {
      overlays.default = final: prev: { };
    };
}
