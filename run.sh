#!/bin/bash

py="main.py"
GPU='0,1'
arch="resnet50"
path="/mount_data/APR_demo"
batch="64"
seg="3"
lr="0.0001"
ep="200"
NOW=`date "+%Y%m%d%H%M"`


for data in 1 2 3 4 5
do
    if [ ${data} -eq 1 ]; then
        train="/mount_data/APR_demo/Datalist/scrambled_eggs/train.txt"
        test="/mount_data/APR_demo/Datalist/scrambled_eggs/test.txt"
        pref="scrambled_eggs"
    elif [ ${data} -eq 2 ]; then
        train="/mount_data/APR_demo/Datalist/tie_tie/train.txt"
        test="/mount_data/APR_demo/Datalist/tie_tie/test.txt"
        pref="tie_tie"
    elif [ ${data} -eq 3 ]; then
        train="/mount_data/APR_demo/Datalist/braid_hair/train.txt"
        test="/mount_data/APR_demo/Datalist/braid_hair/test.txt"
        pref="braid_hair"
    elif [ ${data} -eq 4 ]; then
        train="/mount_data/APR_demo/Datalist/apply_eyeliner/train.txt"
        test="/mount_data/APR_demo/Datalist/apply_eyeliner/test.txt"
        pref="apply_eyeliner"
    else
        train="/mount_data/APR_demo/Datalist/origami/train.txt"
        test="/mount_data/APR_demo/Datalist/origami/test.txt"
        pref="origami"
    fi
    echo "${pref}_train"   
    echo "python3 ${path}/${py} pairwise ${train} ${test} --arch ${arch} --board ${pref} \
    --epochs ${ep} -b ${batch} --lr ${lr} --dropout 0.8 --gpus ${GPU} -j 4 \
    --snapshot_pref ${fail}/best_model/${pref}/${take}_${NOW}_${arch}_${batch}_${ep}"

    python3 ${path}/${py} pairwise ${train} ${test} --arch ${arch} --board ${pref} \
    --epochs ${ep} -b ${batch} --lr ${lr} --dropout 0.8 --gpus ${GPU} -j 4 \
    --snapshot_pref ${fail}/best_model/${pref}/${take}_${NOW}_${arch}_${batch}_${ep}

    python3 ${path}/${py} pairwise ${train} ${test} --arch ${arch} --board ${pref} \
    --epochs ${ep} -b ${batch} --lr ${lr} --dropout 0.8 --gpus ${GPU} -j 4 \
    -e --resume ${fail}/best_model/${pref}/${take}_${NOW}_${arch}_${batch}_${ep}_best_model.pth.tar 
done


echo "end program"
