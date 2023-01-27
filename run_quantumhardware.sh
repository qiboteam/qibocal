#!/bin/bash
#SBATCH --job-name=rbtest
#SBATCH --partition=qpu5q

qq runcards/actions_qq_5q.yml
