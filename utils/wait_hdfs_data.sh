#! /bin/bash
DIR=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

wait_hdfs_data() {
    input=$1
    try_count=0
    constant_count=0
    size=0
    while [ $try_count -le 120 ]
    do
        if $(hadoop fs -test -d $input); then
            hdfs_size=`hadoop fs -du -s $input`
            hdfs_size=`awk -v RS=[0-9]+ '{print RT+0;exit}' <<< "$hdfs_size"`
            if [ $constant_count -eq 0 ]; then
                size=${hdfs_size}
                constant_count=1
                sleep 150s
            else
                if [ "$hdfs_size" -eq "$size" ]; then
                    (( constant_count = constant_count + 1 ))
                    sleep 150s
                else
                    size=${hdfs_size}
                    constant_count=0  # size not constant, reset
                    sleep 150s
                fi
            fi
            if [ $constant_count -gt 5 ]; then
                break
            fi
        else
            echo $input", data not ready, sleep ...", $try_count
            sleep 300s
        fi
        (( try_count = try_count + 1 ))
        if [ $try_count -ge 120 ]; then
            echo $input", data file not ready after n count, n="$count
            if [ $# -eq 2 ]; then
                r_alarm -s "Error: task not ready!!!" -c $input", data file not ready" -u "$2"
            fi
            exit
        fi
    done
    echo $input", data file ready"
}


wait_hdfs_data_onetime() {
    input=$1
    count=0
    while [ $count -le 120 ]
    do
        echo $count
        if $(hadoop fs -test -d $input); then
            break
        else
            (( count = count + 1 ))
            echo $input", data not ready, sleep ...", $count
            sleep 300s
        fi
        if [ $count -ge 120 ]; then
            echo $input", data file not ready after n count, n="$count
            if [ $# -eq 2 ]; then
                r_alarm -s "Error: task not ready!!!" -c $input", data file not ready" -u "$2"
            fi
            exit
        fi
    done
    echo $input", data file ready"
}