#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)


echo "============= download assets for tflite_face_portrait ==============="
cd ${SCRIPT_DIR}/tflite_face_portrait/app/src/main/assets/model/
./download.sh

echo "============= download assets for tflite_dense_depth ==============="
cd ${SCRIPT_DIR}/tflite_dense_depth/app/src/main/assets/model/
./download.sh

echo "============= download assets for tflite_mirnet ==============="
cd ${SCRIPT_DIR}/tflite_mirnet/app/src/main/assets/model/
./download.sh
