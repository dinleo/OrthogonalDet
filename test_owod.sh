#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB or S-OWODB
PORT=${PORT:-"50210"}
set -e

# if raise error, change num_gpus to 1
if [ $BENCHMARK == "M-OWODB" ]; then
  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/t1.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/t2.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/t3.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/t4.pth
else
  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_final.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_final.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_final.pth

  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_final.pth
fi