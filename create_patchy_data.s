#!/bin/bash

source ~/miniconda3/bin/activate base

#set -x

sh_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

src_dir="$(realpath "${sh_dir}/src")"

data_dir="$(realpath "${sh_dir}/data")"

if (( $# != 5 )); then
    echo "Usage: create_patchy_data.s model state SNR ndata runs"
    exit 1
fi

model=$1
state=$2
SNR=$(python3 -c "print('{:1.5f}'.format($3))")
ndata=$(python3 -c "print('{:d}'.format($4))")
runs=$(python3 -c "print('{:d}'.format($5))")

lx=500
ly=500
dx=1
dy=1

npatches=10
patch_r=5


save_dir="${sh_dir}/data/$model/SNR_${SNR}_ndata_${ndata}_state_${state}/npatches_${npatches}_patchr_${patch_r}/lx_${lx}_ly_${ly}/"

if [ ! -d $save_dir ]; then
    mkdir -p $save_dir
fi

params_file="${save_dir}/parameters.json"

echo \
"
{
    "\"model\"" : \"$model\",
    "\"SNR\"" : $SNR,
    "\"ndata\"" : $ndata,
    "\"state\"" : \"$state\",
    "\"npatches\"" : $npatches,
    "\"patch_r\"" : $patch_r,
    "\"lx\"" : $lx,
    "\"ly\"" : $ly,
    "\"dx\"" : $dx,
    "\"dy\"" : $dy
}
" > $params_file

run_start=1
drun=1

run=$(python3 -c "print('{:d}'.format($run_start))")

while (( $(bc <<< "$run <= $runs") ))
do
    run_dir="${sh_dir}/data/$model/SNR_${SNR}_ndata_${ndata}_state_${state}/npatches_${npatches}_patchr_${patch_r}/lx_${lx}_ly_${ly}/run_${run}"

    if [ ! -d $run_dir ]; then
        mkdir -p $run_dir
    fi

    python3 -m models.$model -s $run_dir

    python3 -m analysis.analyse_data -s $run_dir -cdx 5 -cdy 5 -bs 20 -ss 10 -knn 5 -rbf True -v True

    run=$(python3 -c "print('{:d}'.format($run+$drun))")
done