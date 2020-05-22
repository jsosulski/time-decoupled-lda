#!/bin/bash
set -e

if [ -z "$1" ]
  then
    echo "Using last benchmark results stored in last_benchmark_results_short_path.txt as arguments for analysis scripts."
    SHORT_PATH=$(cat last_benchmark_results_short_path.txt)
  else
    SHORT_PATH=$1
fi

source ../../tdlda_venv/bin/activate
cd ..
python main_result_across_datasets.py $SHORT_PATH
python main_moabb_analysis.py $SHORT_PATH