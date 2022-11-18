#!/bin/bash


while true
do 
    python qiboview.py --platform_name tii1q --json_path monitor/monitor_tii1q.json
    python qiboview.py --platform_name tii15q --json_path monitor/monitor_tii5q.json
    python qiboqueue.py 
    sleep 5
done