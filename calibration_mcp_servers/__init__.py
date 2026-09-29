"""MCP servers for qibocal calibration workflows."""

import argparse


def _cmd_assistant(args: argparse.Namespace) -> None:
    from calibration_mcp_servers.assistant_server import assistant_start

    assistant_start()


def _cmd_calibrator(args: argparse.Namespace) -> None:
    from calibration_mcp_servers.automatic_calibration_server import calibrator_start

    calibrator_start()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="qibocal-mcp",
        description="MCP server for submitting quantum circuits to the TII Q-Cloud.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # assistant subcommand
    assistant = sub.add_parser(
        "assistant", help="Start the assistant MCP server (stdio transport)."
    )
    assistant.set_defaults(func=_cmd_assistant)

    # calibrator subcommand
    calibrator = sub.add_parser(
        "calibrator", help="Start the calibrator MCP server (stdio transport)."
    )
    calibrator.set_defaults(func=_cmd_calibrator)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
