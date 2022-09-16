import subprocess
import click
import json
from datetime import datetime



#@click.command()
#@click.option('--platform_name', prompt='Chosen platform', help='The platform you choose.')


#OUT = "monitor/qpu_status.json"
OUT = "qpu_status.json"

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_platform_status(platform_name):

	process = subprocess.getoutput(f"sinfo -p {platform_name} -h")

	return process.split(" ")[-2]


status_qpu1q = get_platform_status("qpu1q")
status_qpu5q = get_platform_status("qpu5q")

data = [[status_qpu1q, status_qpu5q]]
times = [[current_time]]


with open(OUT, "w") as f:
    json.dump({"data" : data, "date-time" : times}, f)
