{
  description = "Julia2Nix development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.nixpkgs.follows = "nixpkgs";
    nixpkgs.url = "github:NixOS/nixpkgs/release-22.05";

    devshell.url = "github:numtide/devshell";
    devshell.inputs.nixpkgs.follows = "nixpkgs";

    julia2nix.url = "github:JuliaCN/Julia2Nix.jl";
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
              pkgs.qt5Full
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
              pkgs.librsvg
              pkgs.libGLU
              pkgs.libGL
              pkgs.xorg.libICE
              pkgs.xorg.libSM
              pkgs.xorg.libXxf86vm
              pkgs.udev
              pkgs.opencl-headers
              pkgs.alsa-lib
              pkgs.xorg.libX11
              pkgs.xorg.libXext
              pkgs.expat
              pkgs.zlib
              pkgs.libuuid
              pkgs.bzip2
              pkgs.libtool
              pkgs.ocl-icd
            ];
            multiPkgs = pkgs: (with pkgs;
              [
                udev
                alsa-lib
              ]);
            runScript = "bash --noprofile --norc";
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
