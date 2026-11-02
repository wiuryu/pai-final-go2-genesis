"""ベースライン: 歩容だけ（開ループ）の指標を、学習済み GRU と同じ条件で測る。"""
import argparse
import json

import torch
import genesis as gs

import walker as W

N_ENV = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="学習済み GRU の重み。指定すると並べて評価する")
    parser.add_argument("--out", default="results/baseline.json")
    parser.add_argument("--seeds", type=int, nargs="*", default=[777, 2024, 4242], help="評価用の初期条件の seed")
    args = parser.parse_args()

    scene, robot = W.build(N_ENV)
    results = {}
    results["open_loop"] = W.evaluate_all(scene, robot, W.OpenLoop(), N_ENV, args.seeds)

    if args.ckpt:
        net = W.Net().to(gs.device)
        net.load_state_dict(torch.load(args.ckpt))
        results["gru"] = W.evaluate_all(scene, robot, net, N_ENV, args.seeds)

    for name, metrics in results.items():
        print(f"{name:10s}", {key: round(value, 3) for key, value in metrics.items()})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
