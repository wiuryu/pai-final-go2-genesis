#!/bin/bash
# 最初から再現する（CPU で約 10 分）。成果物は results/ と models/ に出る。
set -e
mkdir -p results models
python imitate.py --seed 2 --out results/imitate.pt             # 段階1: 教師を DAgger で模倣
python finetune.py --init results/imitate.pt --seed 2 --episodes 20 --w vx=100 --out models/gru_final.pt   # 段階2: シミュレータの勾配で微調整
python baseline.py --ckpt models/gru_final.pt                    # 開ループ歩容との比較指標
python video.py compare --ckpt models/gru_final.pt --out gru_compare.mp4
python video.py inside  --ckpt models/gru_final.pt --out gru_inside.mp4
python video.py summary --ckpt models/gru_final.pt --ckpt1 results/imitate.pt --out gru_summary.mp4
python waypoints.py --ckpt models/gru_final.pt --ckpt1 results/imitate.pt
