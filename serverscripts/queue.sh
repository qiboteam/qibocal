#!/bin/bash
# In order to run the script in background it can be used the command ./queue.sh & 
PLATFORMS_PATH="../../qibolab/src/qibolab/runcards"
SLEEPING_TIME=2
QIBOVIEW_TIME=10 # choose a multiple of SLEEPING_TIME
platformNames=("tii1q" "tii5q")
lastEditDates=()
iterator=0
# Save the last edit date of the platforms' file *.yml
for t in ${platformNames[@]}; do
    lastEditDates+=($(date +%s -r ${PLATFORMS_PATH}/${t}.yml ))
done 
# run an infinite loop 
while true; 
do
    echo $(( iterator % QIBOVIEW_TIME))
    if [[ $(( iterator % QIBOVIEW_TIME)) == 0 ]]; then
        currentPath=$(pwd)
        cd $PLATFORMS_PATH
        git pull 
        cd $currentPath
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
    fi

    python qiboqueue.py 
    sleep $SLEEPING_TIME
    iterator=$(( $iterator + $SLEEPING_TIME ))
    echo $iterator #TODO:remove

done
