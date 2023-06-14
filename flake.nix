{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
  let
    pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
  in
  {
    packages.x86_64-linux = {
      default = pkgs.stdenv.mkDerivation {
        name = "ebsynth";
        src = ./.;
        buildInputs = with pkgs; [
          cudaPackages.cuda_nvcc
          cudatoolkit
        ];
        buildPhase = ''
          ./build-linux-cpu+cuda.sh
        '';
        installPhase = ''
          mkdir -p $out
          mv bin $out
        '';
      };
    };
  };
}
