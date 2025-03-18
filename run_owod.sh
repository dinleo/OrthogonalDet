#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB or S-OWODB
PORT=${PORT:-"50210"}
set -e

if [ $BENCHMARK == "M-OWODB" ]; then
  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/best.pth

  python upload_hf.py
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/t1.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/t2.pth
#
#  python upload_hf.py
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/t2_ft.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/t3.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/t3_ft.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/t4.pth
#
#  python upload_hf.py
else
  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/model_0039999.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/model_0054999.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/model_0069999.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/model_0084999.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/model_0099999.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/model_00114999.pth
fi