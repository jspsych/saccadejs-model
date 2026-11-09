#!/bin/bash
# Full-corpus frame extraction on the Vassar HPC (jr.vassar.edu).
# 4-task array, one emc node each (128 CPUs, 512 GB); subjects are
# partitioned by --shard so tasks never collide. Staged mode: reads
# per-subject tars from /work, writes TFRecords to /work; upload to
# gs://webcam-jspsych-frames happens afterwards from the login node
# (compute nodes could stream to GCS, but staged keeps gcloud/auth
# out of the array job entirely).
#
#   sbatch preprocessing/extract_frames.sbatch
#   tail -f ~/frames-extract-*.log
#
# Afterwards, from the login node:
#   gcloud storage rsync -r /work/jdeleeuw/frames/v1 gs://webcam-jspsych-frames/v1
#   python preprocessing/extract_frames_to_gcs.py --all \
#       --include-list preprocessing/include_list_v1.csv --finalize-manifest
#
# NOTE: must run on the emc partition — the general nodes and login node
# are pre-AVX Opterons that cannot execute tensorflow/mediapipe wheels.
#
#SBATCH --partition=emc
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=frames-extract
#SBATCH --output=/home/jdeleeuw/frames-extract-%a.log

export PYTHONPATH=/home/jdeleeuw/eye-tracking
export TF_CPP_MIN_LOG_LEVEL=3

cd /home/jdeleeuw/eye-tracking
.venv/bin/python preprocessing/extract_frames_to_gcs.py --all \
    --include-list preprocessing/include_list_v1.csv \
    --src /work/jdeleeuw/webm \
    --dst /work/jdeleeuw/frames \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --workers 120
