import os
from glob import glob

import click
import yaml


def folders_exists(folders):
    """Check if a list of folders exists

    Args:
        folders (list): list of absolute or relative path to folders
    """
    foldernames = []
    for foldername in folders:
        expanded = list(glob(foldername))
        if len(expanded) == 0 and "*" not in foldername:
            raise (click.BadParameter(f"file '{foldername}' not found"))
        foldernames.extend(expanded)

    return foldernames


def check_folder_structure(folders):
    """Check if a list of folders share structure between them

    Args:
        folders (list): list of absolute or relative path to folders
    """
    subdirs = [[p for p in Path(folder).glob("**") if p.is_dir()] for folder in folders]
    return all(x == subdirs[0] for x in subdirs)


def update_meta(metadata, metadata_new, target_dir="qq-compare"):
    """Update meta.yml file

    Args:
        metadata (dict): dictionary with the meta.yml actual parameters and values
        metadata_new (dict): dictionary with the new parameters and values to update in the actual meta.yml
    """

    def update(old, new, key):
        old[key] += " , " + new[key]
    
    for key in ("backend", "date", "end-time", "platform", "start-time", "title"):
        update(metadata, metadata_new, key)
    for key in ("numpy", "qibo", "qibocal", "qibolab"):
        update(metadata["versions"], metadata_new["versions"], key)
    with open(f"{target_dir}/meta.yml", "w") as file:
        yaml.safe_dump(metadata, file)


def update_runcard(rundata, rundata_new, target_compare_dir):
    """Update runcard.yml file

    Args:
        rundata (dict): dictionary with the runcard.yml actual parameters and values
        rundata_new (dict): dictionary with the new parameters and values to update in the actual runcard.yml
    """

    rundata["platform"] = rundata["platform"] + " , " + rundata_new["platform"]
    unique = list(set(rundata["qubits"] + rundata_new["qubits"]))
    rundata["qubits"] = unique
    with open(f"{target_compare_dir}/runcard.yml", "w") as file:
        yaml.safe_dump(
            rundata,
            file,
            indent=4,
            allow_unicode=False,
            sort_keys=False,
            default_flow_style=None,
        )
