import subprocess
#import click
import json

#@click.command()
#@click.option('--platform_name', prompt='Chosen platform', help='The platform you choose.')


#OUT = "monitor/qpu_status.json"
OUT = "qpu_status.json"

def get_platform_status(platform_name):
	"""
	Check platform_name's status
	Args:
		platform_name: string which identifies the platform
	Returns: a tuple ["platform_status", integer value representing the jobs in queue]
	"""
	process = subprocess.getoutput(f"sinfo -p {platform_name} -h")
	platform_status = process.split(" ")[-2]

	if(platform_status == "alloc"):
		jobs_on_platform = subprocess.getoutput(f"squeue -p {platform_name} -h")
		len_queue = len(jobs_on_platform.split("\n")) - 1
	else:
		len_queue = 0
		
	return {"name" : platform_name, "status" : platform_status, "queue" : len_queue}


current_time = subprocess.getoutput('''date +"%A %d %B %T" ''')

data_platform_1 = get_platform_status("tii1q")
data_platform_2 = get_platform_status("tii5q")

time = ["Last status check:  " + current_time]

with open(OUT, "w") as f:
    json.dump({"data_platform_1": data_platform_1,
	           "data_platform_2": data_platform_2,
			   "date-time": time}, f)
