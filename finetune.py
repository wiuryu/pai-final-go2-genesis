"""段階2: シミュレータの勾配で GRU を微調整する（truncated BPTT）。

6 秒のエピソードを 20 ステップの窓に区切り、窓ごとに損失を作って scene.backward する。
窓の境界で隠れ状態を切り離す（接触を挟む長い勾配は信頼できないため。README 参照）。
転倒した環境は、以後の損失から外す。
エピソードごとに評価し、最良のモデルを保存する（学習を続けると不安定になるため、これは必須）。
"""
import argparse
import json
import time

import numpy as np
import torch
import genesis as gs

import walker as W


def initial_leg_energy(n_env):
    """脚の動きの大きさ (q - HOME)^2 の移動平均の初期値（歩容の理論値）。"""
    per_leg = [0.0, 0.031, 0.0625]  # hip, thigh, calf
    return torch.tensor(per_leg * 4, device=gs.device).expand(n_env, 12).clone()


def run_episode(scene, robot, net, optimizer, n_env, gen):
    """1 エピソード分、窓ごとに勾配を計算してパラメータを更新する。"""
    W.reset_episode(scene, robot, n_env, gen)
    h = net.init_h(n_env)
    alive = torch.ones(n_env, device=gs.device)  # 転倒した環境は 0 にして損失から外す
    leg_energy = initial_leg_energy(n_env)
    foot_prev = None

    for window in range(W.T_EP // W.WIN):
        window_loss = 0.0

        for k in range(W.WIN):
            t = (window * W.WIN + k) * W.DT
            state, _, h = W.step_policy(scene, robot, net, h, t)
            alive = alive * (state.qpos[:, 2] > 0.2).float().detach()

            # 脚の動きの大きさの移動平均（左右対称の罰に使う）
            joint_dev2 = (state.qpos[:, 7:] - W.home()) ** 2
            leg_energy = W.LAM * leg_energy + (1 - W.LAM) * joint_dev2

            # 足の滑り（接地中の足先の水平速度の二乗）
            foot = W.foot_pos(state)
            slip = 0.0
            if foot_prev is not None:
                contact, slip2 = W.contact_slip(foot, foot_prev)
                slip = (contact * slip2).sum(dim=1)
            foot_prev = foot

            step_cost = (
                W.step_loss(state)
                + W.LOSS_WEIGHT["sym"] * W.sym_pen(leg_energy)
                + W.LOSS_WEIGHT["slip"] * slip
            )
            window_loss = window_loss + (step_cost * alive).sum() / n_env / W.WIN

        optimizer.zero_grad()
        scene.backward(window_loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()

        # 窓の境界で勾配を切る
        h = h.detach()
        leg_energy = leg_energy.detach()
        foot_prev = foot_prev.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True, help="段階1 で保存した重み")
    parser.add_argument("--net", default="gru")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--nenv", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="results/finetune.pt")
    parser.add_argument("--w", nargs="*", default=[], help="損失の重みの上書き。例: --w slip=2 vx=60")
    args = parser.parse_args()

    for item in args.w:
        name, value = item.split("=")
        W.LOSS_WEIGHT[name] = float(value)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)

    scene, robot = W.build(args.nenv)
    net = W.Net(args.net).to(gs.device)
    net.load_state_dict(torch.load(args.init))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(1000 + args.seed)

    # 勾配モードの状態バッファを、評価（no_grad）より先に作っておく
    W.reset_episode(scene, robot, args.nenv, gen)
    W.step_policy(scene, robot, net, net.init_h(args.nenv), 0.0)

    metrics = W.evaluate(scene, robot, net, args.nenv)
    best_score = W.score(metrics)
    best_episode = 0
    best_weights = {k: v.clone() for k, v in net.state_dict().items()}
    start = time.time()
    print(f"ep 0 score {best_score:.3f} yaw {metrics['yaw_abs']:.3f} slip {metrics['slip']:.3f}", flush=True)

    for episode in range(1, args.episodes + 1):
        run_episode(scene, robot, net, optimizer, args.nenv, gen)

        metrics = W.evaluate(scene, robot, net, args.nenv)
        score = W.score(metrics)
        if score < best_score:
            best_score, best_episode = score, episode
            best_weights = {k: v.clone() for k, v in net.state_dict().items()}
        print(
            f"ep {episode}/{args.episodes} score {score:.3f} fall {metrics['fall_rate']:.2f} "
            f"yaw {metrics['yaw_abs']:.3f} slip {metrics['slip']:.3f} t {time.time() - start:.0f}s",
            flush=True,
        )

    net.load_state_dict(best_weights)
    torch.save(net.state_dict(), args.out)

    result = W.evaluate_all(scene, robot, net, args.nenv)
    print("best ep", best_episode, "test", {k: round(v, 3) for k, v in result.items()}, flush=True)
    summary = {"best_ep": best_episode, "test": result, "args": vars(args)}
    with open(args.out.replace(".pt", ".json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
