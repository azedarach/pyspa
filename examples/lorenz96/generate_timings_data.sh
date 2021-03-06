#!/bin/bash

model="lorenz96"

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

forcing_types="constant periodic"
forcings="3 5.3 7 10"
time_step="0.01"

for t in $forcing_types ; do
    prefix="${model}_${t}_forcing"
    input_suffix="_time_step-${time_step}.csv"
    output_suffix="_time_step-${time_step}_timings.csv"
    for f in $forcings ; do
        data_files=$(ls $input_dir/${prefix}_F-${f}_length-*${input_suffix})
        output_file="${output_dir}/${prefix}_F-${f}${output_suffix}"
        $PYTHON $run_timings \
                --min-clusters $min_clusters \
                --max-clusters $max_clusters \
                --discard-fraction $discard_fraction \
                --output-file "$output_file" \
                $data_files
    done
done
