"""段階1: 教師（歩容＋ヨー P 制御）を DAgger で GRU に模倣させる。

教師は「歩容に、ヨーを戻す向きの thigh 振幅の左右差を足したもの」。
GRU には速度を含まない 17 次元だけを見せる。

DAgger の流れ:
  1. 教師の走行データで GRU を学習する
  2. 学習した GRU を走らせ、通った状態のすべてに教師の正解ラベルを付けて追加する
  3. 1 に戻る（これを繰り返す）
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import genesis as gs

import walker as W

KP_YAW = 0.5  # 教師のヨーゲイン


def expert_target(state, t):
    """教師の目標角（HOME からの差）。

    歩容に加えて、ヨーに比例して thigh の振幅に左右差をつける。
    左脚の振幅を増やし右脚を減らす（またはその逆）と、進行方向が変わる。
    """
    dtype, device = gs.tc_float, gs.device
    yaw = W.rpy(state.qpos[:, 3:7])[2]
    phase = torch.tensor(W.PHASE, dtype=dtype, device=device)
    side = torch.tensor(W.SIDE, dtype=dtype, device=device)
    leg_sin = torch.sin(2 * np.pi * (W.FREQ * t + phase))  # 脚ごとの位相（4 脚）

    target = W.gait_offset(t).expand(yaw.shape[0], 12).clone()
    steering = KP_YAW * yaw  # (B,)
    target[:, 1::3] += W.A_TH * steering[:, None] * side * leg_sin
    return target


@torch.no_grad()
def collect(scene, robot, net, n_env, gen, beta, noise):
    """1 エピソード走らせて、(観測, 教師の正解) の系列を集める。

    net=None なら教師の行動で走る。
    そうでなければ、各環境・各ステップで確率 beta で教師、それ以外は GRU の行動で走る。
    どちらの場合も、正解ラベルは常に教師の目標角。
    """
    W.reset_episode(scene, robot, n_env, gen)
    h = net.init_h(n_env) if net is not None else None
    observations, labels = [], []

    for i in range(W.T_EP):
        t = i * W.DT
        state = scene.sim.rigid_solver.get_state()
        expert_action = expert_target(state, t)
        obs = W.observe(state, t)
        observations.append(obs)
        labels.append(expert_action)

        if net is None:
            action = expert_action
        else:
            net_action, h = net(obs, h)
            use_expert = torch.rand(n_env, 1, generator=gen) < beta
            use_expert = use_expert.to(gs.device).to(expert_action.dtype)
            action = use_expert * expert_action + (1 - use_expert) * net_action

        action = action + noise * torch.randn(action.shape, generator=gen).to(action)
        W.apply_target(robot, state, W.home() + action)
        scene.step()

    return torch.stack(observations, dim=1), torch.stack(labels, dim=1)  # (B, T, 17), (B, T, 12)


def fit(net, observations, labels, steps, lr, batch_size=64):
    """集めたデータで、GRU の出力が教師の目標角に近づくように学習する。最後の損失を返す。"""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    n_episodes = observations.shape[0]
    loss = None

    for _ in range(steps):
        idx = torch.randint(0, n_episodes, (min(batch_size, n_episodes),))
        prediction = net.sequence(observations[idx])
        loss = ((prediction - labels[idx]) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="gru")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=6, help="DAgger の繰り返し回数")
    parser.add_argument("--steps", type=int, default=300, help="1 回あたりの学習ステップ数")
    parser.add_argument("--nenv", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--out", default="results/imitate.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)

    scene, robot = W.build(args.nenv)
    gen = torch.Generator().manual_seed(100 + args.seed)
    net = W.Net(args.net).to(gs.device)
    start = time.time()

    # 最初のデータは、雑音をつけた教師自身の走行
    observations, labels = collect(scene, robot, None, args.nenv, gen, beta=1.0, noise=0.03)

    best_score, best_iter, best_weights = float("inf"), -1, None
    for it in range(args.iters):
        loss = fit(net, observations, labels, args.steps, args.lr)

        metrics = W.evaluate(scene, robot, net, args.nenv, seed=1234)
        score = metrics["yaw_abs"] + metrics["vx_err"] + 2 * metrics["fall_rate"]
        if score < best_score:
            best_score, best_iter = score, it
            best_weights = {k: v.clone() for k, v in net.state_dict().items()}
        print(
            f"it {it} loss {loss:.5f} score {score:.3f} fall {metrics['fall_rate']:.2f} "
            f"yaw {metrics['yaw_abs']:.3f} vx {metrics['vx_mean']:.3f} t {time.time() - start:.0f}s",
            flush=True,
        )

        # 学習した GRU の走行に教師のラベルを付けて追加する（最初の 1 回だけ 30% 教師を混ぜる）
        beta = 0.3 if it == 0 else 0.0
        new_obs, new_labels = collect(scene, robot, net, args.nenv, gen, beta=beta, noise=0.0)
        observations = torch.cat([observations, new_obs])
        labels = torch.cat([labels, new_labels])

    net.load_state_dict(best_weights)
    torch.save(net.state_dict(), args.out)
    print("best it", best_iter, "test", W.evaluate_all(scene, robot, net, args.nenv), flush=True)


if __name__ == "__main__":
    main()
