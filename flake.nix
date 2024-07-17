{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devenv = {
      url = "github:cachix/devenv";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    flake-parts,
    ...
  } @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;} {
      imports = [inputs.devenv.flakeModule];
      systems = ["x86_64-linux" "aarch64-darwin"];

      perSystem = {pkgs, ...}: {
        packages.default = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = self;
          preferWheels = true;
        };

        devenv.shells.default = {
          packages = with pkgs; [poethepoet pre-commit stdenv.cc.cc.lib];

          languages = {
            python = {
              enable = true;
              poetry = {
                enable = true;
                install = {
                  enable = true;
                  allExtras = true;
                  groups = ["dev" "test"];
                };
              };
            };
          };
        };
      };
    };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };
}
