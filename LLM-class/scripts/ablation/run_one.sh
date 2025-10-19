#!/usr/bin/env bash
# usage: ./run_one.sh <run_name> <extra_args...>
RUN_NAME=$1
shift
OUTDIR=results/ablation/${RUN_NAME}
mkdir -p ${OUTDIR}
echo "Running ${RUN_NAME} -> ${OUTDIR}"
# run train and save logs
python src/train.py "$@" --save_dir ${OUTDIR} 2>&1 | tee ${OUTDIR}/log.txt
