#!/usr/bin/env python
import argparse

import curio
import inotify.adapters
import inotify.constants
from curio import subprocess


async def main(folder, exe_args):
    i = inotify.adapters.Inotify()
    i.add_watch(folder)

    for event in i.event_gen(yield_nones=False):
        if event is not None:
            (header, _, _, _) = event
            if (
                (header.mask & inotify.constants.IN_CREATE)
                or (header.mask & inotify.constants.IN_DELETE)
                or (header.mask & inotify.constants.IN_MODIFY)
            ):
                await subprocess.run(exe_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("exe_args", nargs="+")
    args = parser.parse_args()
    curio.run(main(args.folder, args.exe_args))
