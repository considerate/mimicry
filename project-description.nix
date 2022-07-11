{ stdenv, texlive }:
stdenv.mkDerivation {
  name = "mimicry-project-description";
  src = ./.;
  makeFlags = "prefix=$(out)";
  nativeBuildInputs = [
    texlive.combined.scheme-medium
  ];
}
