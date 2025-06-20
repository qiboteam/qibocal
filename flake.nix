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
        devenv.shells.default = {config, ...}: {
          packages = with pkgs; [poethepoet pre-commit stdenv.cc.cc.lib];

          env = {
            QIBOLAB_PLATFORMS = (dirOf config.env.DEVENV_ROOT) + "/qibolab_platforms_qrc";
          };

          languages = {
            python = {
              enable = true;
              libraries = with pkgs; [zlib];
              poetry = {
                enable = true;
                install = {
                  enable = true;
                  allExtras = true;
                  groups = ["dev" "tests"];
                };
              };
            };
          };
        };
      };
    };
}
