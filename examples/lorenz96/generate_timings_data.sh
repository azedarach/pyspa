#!/bin/bash

model="lorenz96"
forcing_type="constant"
prefix="${model}_${forcing_type}_forcing"

PYTHON="python"
run_timings="time_${model}_training.py"

min_clusters="2"
max_clusters="10"
discard_fraction="0.2"

input_dir="input-data"
output_dir="output-data"

if test ! -d $input_dir ; then
    echo "Error: input data directory not found"
    exit 1
fi

if test ! -d $output_dir ; then
    mkdir $output_dir
fi

forcings="3 5.3 7 10"
time_step="0.01"

for f in $forcings ; do
    data_files=$(ls $input_dir/${prefix}_F-${f}_length-*_time_step-${time_step}.csv)
    output_file="${output_dir}/${prefix}_F-${f}_time_step-${time_step}_timings.csv"
    $PYTHON $run_timings \
            --min-clusters $min_clusters \
            --max-clusters $max_clusters \
            --discard-fraction $discard_fraction \
            --output-file "$output_file" \
            $data_files
done
