#!/bin/bash
# Headless setup for a fresh google-colab-cli LEAN TPU VM (colab --tpu v5e1).
# Turns a bare VM into a training-ready one with NO interactive steps, using the
# scoped read-only service account (objectViewer on gs://webcam-jspsych-frames).
# Established 2026-06-15; see requirements-colab.txt + colab-tpu-workflow notes.
#
# From LOCAL first (control-plane uploads):
#   S=cscreen
#   colab --auth=adc new -s $S --tpu v5e1
#   colab --auth=adc upload -s $S requirements-colab.txt              /content/requirements-colab.txt
#   colab --auth=adc upload -s $S ~/.config/gcloud/keys/webcam-colab-reader.json /content/sa.json
#   tar czf /tmp/etcode.tar.gz et_util analysis/*.py *.md
#   colab --auth=adc upload -s $S /tmp/etcode.tar.gz                  /content/etcode.tar.gz
#   colab --auth=adc upload -s $S training/colab_bootstrap.sh         /content/colab_bootstrap.sh
# Then run THIS detached (frees the kernel; never import jax in-kernel or the
# training subprocess loses the TPU):
#   echo 'import subprocess; subprocess.Popen("bash /content/colab_bootstrap.sh > /content/bootstrap.log 2>&1 &", shell=True)' \
#     | colab --auth=adc exec -s $S
#   # poll: colab --auth=adc exec -s $S  <<< 'import subprocess;print(subprocess.run("tail -3 /content/bootstrap.log",shell=True,capture_output=True,text=True).stdout)'
set -uo pipefail
cd /content

echo "=== 1/4 GCS auth (scoped read-only SA) ==="
gcloud auth activate-service-account --key-file=/content/sa.json

echo "=== 2/4 deps (lean image lacks tensorflow; pins avoid the 2.20 wheel + pkg_resources traps) ==="
pip install -q -r /content/requirements-colab.txt

echo "=== 3/4 code ==="
rm -rf code && mkdir code && tar xzf etcode.tar.gz -C code

echo "=== 4/4 data (~7 GB, GCP-internal transfer) ==="
mkdir -p v1_train_repack
gcloud storage rsync -r gs://webcam-jspsych-frames/v1_train_repack v1_train_repack

echo "BOOTSTRAP_DONE  data: $(ls v1_train_repack 2>/dev/null | tr '\n' ' ')"
