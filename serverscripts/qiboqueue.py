import subprocess
import json
import os
import pandas as pd

OUT = "running_jobs.json"

#subprocess.call("", shell=True)
p = subprocess.getoutput("echo ciao")
print(p)
#retcode = p.wait()

#df = pd.read_csv('running_jobs.csv')
#df_tii1q = df[]


#with open(OUT, "w") as f:
#    json.dump({"data": data}, f)