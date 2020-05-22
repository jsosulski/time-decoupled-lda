#!/bin/bash
set -e
echo "Results run name" $RESULTS_RUN_NAME
cd ..
source ../tdlda_venv/bin/activate
python main_moabb_pipeline.py "$@"
