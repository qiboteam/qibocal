# -*- coding: utf-8 -*-
import click
import json
from qibolab import Platform
import sys

# insertion of platform name and path to *json file
@click.command()
@click.option('--platform_name', prompt='Chosen platform', help='The platform you choose.')
@click.option('--json_path', prompt='*.json path', help='Where you want to save the *.json')


def main(platform_name, json_path):

    if (platform_name != "tii1q" and platform_name != "tii5q"): 
        print("Please select one platform between tii1q and tii5q")
        sys.exit()

    OUT_MONITOR = json_path

    REQUIRED_FILE_METADATA = {"resonator_freq", "qubit_freq", "T1", "T2"}

    p = Platform(platform_name)
    platform_data = p.settings
    nqubits = platform_data["nqubits"]
    
    data = []
    for _ in range(nqubits):
        info_qubit = [_ + 1]+[platform_data["characterization"]["single_qubit"][_][i] for i in REQUIRED_FILE_METADATA] 
        data.append(info_qubit)

    with open(OUT_MONITOR, "w") as f:
        json.dump({"data": data}, f)


if __name__ == "__main__":
    main()