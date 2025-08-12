#!/bin/bash

source ~/miniconda3/bin/activate base

#set -x

sh_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

src_dir="$(realpath "${sh_dir}/src")"

data_dir="$(realpath "${sh_dir}/data")"

if (( $# != 6 )); then
    echo "Usage: analyze_patchy_data.s model state SNR ndata npatches runs"
    exit 1
fi

model=$1
state=$2
SNR=$(python3 -c "print('{:d}'.format($3))")
ndata=$(python3 -c "print('{:d}'.format($4))")
npatches=$(python3 -c "print('{:d}'.format($5))")
runs=$(python3 -c "print('{:d}'.format($6))")

lx=500
ly=500
dx=1
dy=1

patch_r=15

save_dir="${sh_dir}/data/$model/SNR_${SNR}_ndata_${ndata}_state_${state}/npatches_${npatches}_patchr_${patch_r}/lx_${lx}_ly_${ly}/"

run_start=1
drun=1

run=$(python3 -c "print('{:d}'.format($run_start))")

while (( $(bc <<< "$run <= $runs") ))
do
    run_dir="${sh_dir}/data/$model/SNR_${SNR}_ndata_${ndata}_state_${state}/npatches_${npatches}_patchr_${patch_r}/lx_${lx}_ly_${ly}/run_${run}"

    python3 -m analysis.analyse_data -s $run_dir -cdx 5 -cdy 5 -bs 20 -ss 10 -knn 5 -rbf True -v True

    run=$(python3 -c "print('{:d}'.format($run+$drun))")
done