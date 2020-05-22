#!/bin/bash
set -eu

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

# read yaml config
eval $(parse_yaml ../local_config.yaml conf_)

RESULTS_ROOT=${conf_results_root}
BENCHMARK_META_NAME=${conf_benchmark_meta_name}
RESULTS_RUN_NAME=$(date +%Y-%m-%d)
RESULTS_GROUPING=$(uuidgen)

echo "Results root:        " ${RESULTS_ROOT}
echo "Benchmark meta name: " ${BENCHMARK_META_NAME}
echo "Results run name:    " ${RESULTS_RUN_NAME}
echo "Results grouping:    " ${RESULTS_GROUPING}

RESULTS_FOLDER="${RESULTS_ROOT}/${BENCHMARK_META_NAME}/${RESULTS_RUN_NAME}/${RESULTS_GROUPING}"

echo "Results will be stored in: "
echo "    ${RESULTS_FOLDER}"

echo "Storing a snapshot of used analysis_config.yaml in results folder."
mkdir -p ${RESULTS_FOLDER}
cp ../analysis_config.yaml ${RESULTS_FOLDER}/
echo ${RESULTS_FOLDER} > last_benchmark_results_path.txt
echo ${RESULTS_RUN_NAME}/${RESULTS_GROUPING} > last_benchmark_results_short_path.txt

#TARGET_ENV="RESULTS_RUN_NAME=${RESULTS_RUN_NAME}"
#TARGET_ENV="${TARGET_ENV},RESULTS_GROUPING=${RESULTS_GROUPING}"

export RESULTS_RUN_NAME
export RESULTS_GROUPING

# These commands are used on MOAB / TORQUE managed clusters
#DEBUG_CMD="msub -q express -l nodes=1:ppn=4,pmem=6gb,walltime=10:00 -d . -v $TARGET_ENV ./run_single_job.sh -F"
#PRODUCTION_CMD="msub -l nodes=1:ppn=4,pmem=6gb,walltime=0:30:00:00 -d . -v $TARGET_ENV ./run_single_job.sh -F"

PRODUCTION_CMD="./run_single_job.sh"

echo "Using following command for submission:"
echo $PRODUCTION_CMD
MSUB_CMD=${PRODUCTION_CMD}

SUBMISSION_ARGS_FILE=$1

# Target env not working

cat ${SUBMISSION_ARGS_FILE} | while read line
do
   echo "submitting ${line}"
   echo ${MSUB_CMD} "${line}"
   [[ !  -z  ${line}  ]] && ${MSUB_CMD} "${line}"
done
