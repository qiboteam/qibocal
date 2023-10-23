"""qibocal-index-reports.py
Generates a JSON index with reports information.
"""
import json
import pathlib
import sys
from collections import ChainMap

ROOT = "/home/users/qibocal/qibocal-reports"
ROOT_URL = "http://login.qrccluster.com:9000/"
OUT = "/home/users/qibocal/qibocal-reports/index.json"
DEFAULTS = {
    "title": "-",
    "date": "-",
    "platform": "-",
    "start-time": "-",
    "end-time": "-",
    "tag": "-",
}
REQUIRED_FILE_METADATA = {"title", "date", "platform", "start-time" "end-time", "tag"}


def meta_from_path(p):
    meta = ChainMap(DEFAULTS)
    json_meta = p / "meta.json"
    json_res = {}
    if json_meta.exists():
        with json_meta.open() as f:
            try:
                json_res = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print(f"Error processing {json_meta}: {e}", file=sys.stderr)
    meta = meta.new_child(json_res)
    return meta


def register(p):
    path_meta = meta_from_path(p)
    title, date, platform, start_time, end_time, tag = (
        path_meta["title"],
        path_meta["date"],
        path_meta["platform"],
        path_meta["start-time"],
        path_meta["end-time"],
        path_meta["tag"],
    )
    url = ROOT_URL + p.name
    titlelink = f'<a href="{url}">{title}</a>'
    return (titlelink, date, platform, start_time, end_time, tag)


def make_index():
    root_path = pathlib.Path(ROOT)
    data = []
    for p in root_path.iterdir():
        if p.is_dir():
            try:
                res = register(p)
                data.append(res)
            except:
                print("Error processing folder", p, file=sys.stderr)
                raise

    with open(OUT, "w") as f:
        json.dump({"data": data}, f)


if __name__ == "__main__":
    make_index()
