{
  config,
  pkgs,
  ...
}:
{
  packages = with pkgs; [
    poethepoet
    pre-commit
    stdenv.cc.cc.lib
  ];

  env = {
    QIBOLAB_PLATFORMS = (dirOf config.env.DEVENV_ROOT) + "/qibolab_platforms_qrc";
    PYTHONBREAKPOINT = "pudb.set_trace";
  };

  languages.python = {
    enable = true;
    venv.enable = true;
    version = "3.12";
    uv = {
      enable = true;
      sync = {
        enable = true;
        allGroups = true;
        allExtras = true;
      };
    };
  };
}
