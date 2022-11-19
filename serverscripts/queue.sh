#!/bin/bash
# In order to run the script in background it can be used the command ./queue.sh & 
PLATFORMS_PATH="../../qibolab/src/qibolab/runcards"
platformNames=("tii1q" "tii5q")
lastEditDates=()
# Save the last edit date of the platforms' file *.yml
for t in ${platformNames[@]}; do
    lastEditDates+=($(date +%s -r ${PLATFORMS_PATH}/${t}.yml ))
done 
# run an infinite loop 
while true; 
do 
    
    for i in ${!platformNames[@]}; do
        # date in second of the last edit
        newLastEditDate=$(date +%s -r ${PLATFORMS_PATH}/${platformNames[$i]}.yml )
        echo ${platformNames[i]} #TODO:remove 
        echo $newLastEditDate #TODO:remove 
        echo ${lastEditDates[$i]} #TODO:remove 
        # if the file's edit date is later than lastEditDate, the script update the *json file
        if [[ ${lastEditDates[$i]} < $newLastEditDate ]]; then
            python qiboview.py --platform_name ${platformNames[i]} --json_path monitor/monitor_${platformNames[i]}.json
            lastEditDates[$i]=$newLastEditDate
        fi 
        
    done

    python qiboqueue.py 
    sleep 2

done
