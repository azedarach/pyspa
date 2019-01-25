#!/bin/bash

model="lorenz96"
forcing_type="constant"

PYTHON="python"
run_cmd="run_${model}_system.py"

data_dir="input-data"

prefix="${model}_${forcing_type}_forcing"

forcing_amps="3 5.3 7 10"
lengths="100 500 1000 3000 10000"
#"100 500 1000 3000 10000 50000 100000 1000000"
time_step="0.01"

if test ! -d "$data_dir" ; then
    mkdir "$data_dir"
fi

for a in $forcing_amps ; do
    for l in $lengths ; do
        output_file="${data_dir}/${prefix}_F-${a}_length-${l}_time_step-${time_step}.csv"
        $PYTHON $run_cmd \
                --forcing-type $forcing_type \
                --forcing-amp $a \
                --length $l \
                --time-step $time_step \
                --output-file "$output_file"
    done
done
