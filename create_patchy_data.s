#!/bin/bash

source ~/miniconda3/bin/activate base

#set -x

sh_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

src_dir="$(realpath "${sh_dir}/src")"

data_dir="$(realpath "${sh_dir}/data")"

if (( $# != 5 )); then
    echo "Usage: create_patchy_data.s model state SNR ndata run"
    exit 1
fi

model=$1
state=$2
SNR=$(python3 -c "print('{:1.5f}'.format($3))")
ndata=$(python3 -c "print('{:d}'.format($4))")
run=$(python3 -c "print('{:d}'.format($5))")

lx=500
ly=500
dx=1
dy=1

npatches=10
patch_r=5

save_dir="${sh_dir}/data/$model/SNR_${SNR}_ndata_${ndata}_state_${state}/npatches_${npatches}_patchr_${patch_r}/lx_${lx}_ly_${ly}/run_${run}"

if [ ! -d $save_dir ]; then
    mkdir -p $save_dir
fi

params_file="${save_dir}/parameters.json"


echo \
"
{
    "\"model\"" : \"$model\",
    "\"run\"" : $run,
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

python3 -m models.$model -s $save_dir