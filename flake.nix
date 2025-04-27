{
  description = "A Nix-flake-based Python development environment";

  inputs = {
    nixpkgs = {
      url = "github:NixOS/nixpkgs/nixos-24.11";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              overlays = [
              ];
            };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        let
          pythonPackages = pkgs.python312Packages;
        in
        {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              pythonPackages.numpy
              pythonPackages.pygame_sdl2
              pythonPackages.termcolor

              # Check
              pythonPackages.autopep8
              mypy
              pylint
              pythonPackages.pytest
            ];

            packages = with pkgs; [
              # graphviz
              pythonPackages.ipython
              pythonPackages.importmagic
              # epc
              pylyzer
              ruff
            ];

            env = {
            };
          };
        }
      );
    };
}
