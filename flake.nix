{
  description = "Mimicry";
  inputs = {
    nixpkgs = {
      url = "github:NixOS/nixpkgs/release-22.05";
    };
  };

  outputs = { self, nixpkgs }: {
    overlays = {
      build-pdf = final: prev: {
        project-description = final.callPackage ./project-description.nix { };
      };
    };

    packages.x86_64-linux.project-description =
      let pkgs = import nixpkgs {
        system = "x86_64-linux";
        overlays = [
          self.overlays.build-pdf
        ];
      };
      in pkgs.project-description;
    packages.x86_64-linux.default = self.packages.x86_64-linux.project-description;
    defaultPackage.x86_64-linux = self.packages.x86_64-linux.default;
  };
}
