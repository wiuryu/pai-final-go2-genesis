"""学習済み GRU を、経路上の点を順に通過するタスクに流用する（再学習なし）。

GRU は「ヨーを 0 に戻す」ように学習してある。入力のヨーを「目標方向とのずれ」に置き換えると、
GRU は目標方向へ向きを変えて歩く。目標方向は、現在位置から次の点へ向かう方向とする。
"""
import argparse
import json

import numpy as np
import torch
import genesis as gs

import walker as W

# 経路上の点（出発点からの x, y [m]）。x 方向に 2 m ずつ進みながら、左右にゆるやかに曲がる
WAYPOINTS = [(2, 0.6), (4, 1.2), (6, 0.6), (8, -0.4), (10, -1.2), (12, -0.6), (14, 0.4), (16, 1.0)]
REACH_RADIUS = 0.4    # 点からこの距離以内に入ったら通過とみなす [m]
MAX_TURN = 0.6        # GRU に見せるヨーのずれの上限 [rad]。学習時の初期ヨーの範囲に合わせる
T_MAX = 45.0          # 最大の時間 [s]
YAW_INDEX = 16        # 観測 17 次元のうち、ヨーの位置


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def steered_observation(state, t, target_heading):
    """GRU への観測。ヨーの代わりに「ヨー - 目標方向」を入れる。"""
    obs = W.observe(state, t).clone()
    yaw = W.rpy(state.qpos[:, 3:7])[2]
    error = wrap_angle(yaw - target_heading)
    obs[:, YAW_INDEX] = torch.clamp(error, -MAX_TURN, MAX_TURN)
    return obs


@torch.no_grad()
def follow_waypoints(scene, robot, net, origin, waypoints, on_step=None, index=0):
    """出発点 origin（x, y）から、waypoints を順に通過するように歩かせる。

    on_step(t, state, 次の点の番号, 目標方向) があれば、毎ステップ呼ぶ（映像用）。
    index は、複数台のロボットがいるシーンで、robot が何台目か（0 始まり）を表す。
    各点の通過時刻（通過できなかった点は None）と、軌跡を返す。
    """
    gen = torch.Generator().manual_seed(0)
    W.reset_episode(scene, robot, 1, gen, yaw_range=0.0)
    h = net.init_h(1)
    targets = [(origin[0] + x, origin[1] + y) for x, y in waypoints]
    passed = [None] * len(targets)
    next_index, track = 0, []

    for i in range(int(T_MAX / W.DT)):
        t = i * W.DT
        state = W.RobotState(scene.sim.rigid_solver.get_state(), index)
        x, y = state.qpos[0, 0].item(), state.qpos[0, 1].item()
        track.append((x, y))

        if next_index < len(targets):
            goal = targets[next_index]
            if np.hypot(goal[0] - x, goal[1] - y) < REACH_RADIUS:
                passed[next_index] = t
                next_index += 1
        if next_index >= len(targets):
            break

        goal = targets[next_index]
        heading = np.arctan2(goal[1] - y, goal[0] - x)
        if on_step is not None:
            on_step(t, state, next_index, heading)

        obs = steered_observation(state, t, torch.tensor([heading], dtype=gs.tc_float, device=gs.device))
        action, h = net(obs, h)
        W.apply_target(robot, state, W.home() + action)
        scene.step()
        if state.qpos[0, 2].item() < 0.18:
            break  # 転倒

    return passed, np.array(track)


# 評価に使う経路。出発点からの (x, y) [m]
PATHS = {
    "元の経路": WAYPOINTS,
    "左右反転": [(x, -y) for x, y in WAYPOINTS],
    "大きく蛇行": [(2, 1.0), (4, -0.2), (6, 1.0), (8, -0.2), (10, 1.0), (12, -0.2)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/gru_final.pt", help="学習段階 2（最終）の重み")
    parser.add_argument("--ckpt1", default="results/imitate_final.pt", help="比較用: 学習段階 1 の重み")
    parser.add_argument("--out", default="results/waypoints.json")
    args = parser.parse_args()

    scene, robot = W.build(1)
    results = {}
    for name, ckpt in [("学習段階 2（最終）", args.ckpt), ("学習段階 1（模倣のみ）", args.ckpt1)]:
        net = W.Net().to(gs.device)
        net.load_state_dict(torch.load(ckpt))
        results[name] = {}
        for path_name, waypoints in PATHS.items():
            passed, track = follow_waypoints(scene, robot, net, (0.0, 0.0), waypoints)
            n_passed = sum(p is not None for p in passed)
            results[name][path_name] = {"passed": n_passed, "total": len(waypoints), "seconds": round(len(track) * W.DT, 1)}
            print(f"{name} / {path_name}: {n_passed} / {len(waypoints)} 点を通過（{len(track) * W.DT:.1f} s）", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
