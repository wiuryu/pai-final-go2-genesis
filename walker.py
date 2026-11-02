"""Go2 四脚ロボットを Genesis（微分可能シミュレータ）で歩かせるための共通部品。

環境・歩容・観測・GRU・損失・評価をここにまとめる。
課題: 初期ヨー（進行方向）が ±0.6 rad ずれた状態から出発し、ヨーを 0 に戻して前進速度 0.5 m/s で歩く。
"""
import numpy as np
import torch
import torch.nn as nn
import genesis as gs

# ---- シミュレーション ----
DT = 0.01       # 刻み 10 ms
T_EP = 600      # 1 エピソードのステップ数（6 秒）
WIN = 20        # 勾配を流す窓の長さ（ステップ）
KP = 80.0       # 関節 PD 制御のゲイン。トルクを自前で計算する（勾配を通すため）
KV = 2.0
V_TGT = 0.5     # 目標前進速度 [m/s]
DOFS = list(range(6, 18))      # ベースの 6 自由度の後ろにある 12 関節
HOME = [0.0, 0.8, -1.5] * 4    # 立位の関節角。順序は FL, FR, RL, RR それぞれの (hip, thigh, calf)

# ---- 手書き歩容（トロット）: 探索で決めた値 ----
FREQ = 2.5                     # 周波数 [Hz]
A_TH = 0.20                    # thigh の振幅 [rad]
A_CF = 0.30                    # calf の振幅 [rad]
A_HP = -0.10                   # hip の振幅 [rad]
PHASE = [0.0, 0.5, 0.5, 0.0]   # FL+RR と FR+RL が逆位相
SIDE = [1.0, -1.0, 1.0, -1.0]  # 左 +1 / 右 -1


def build(n_env):
    """シミュレーションを作り、(scene, robot) を返す。"""
    gs.init(backend=gs.cpu, logging_level="error")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=DT, requires_grad=True),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.42)))
    scene.build(n_envs=n_env)
    return scene, robot


def home():
    return torch.tensor(HOME, dtype=gs.tc_float, device=gs.device)


def gait_offset(t):
    """歩容による HOME からの目標角のずれ（12 次元）。"""
    dtype, device = gs.tc_float, gs.device
    phase = torch.tensor(PHASE, dtype=dtype, device=device)
    side = torch.tensor(SIDE, dtype=dtype, device=device)
    s = torch.sin(2 * np.pi * (FREQ * t + phase))    # 脚ごとの位相（4 脚）

    offset = torch.zeros(12, dtype=dtype, device=device)
    offset[0::3] = A_HP * side * s                   # hip: 左右で逆符号に外転
    offset[1::3] = A_TH * s                          # thigh: 前後に振る
    offset[2::3] = -A_CF * torch.clamp(-s, min=0.0)  # calf: 遊脚相に畳む
    return offset


def rpy(quat):
    """クォータニオン (w, x, y, z) から roll, pitch, yaw を求める。"""
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1 + 1e-6, 1 - 1e-6))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def observe(state, t):
    """GRU への入力（17 次元）。速度は含めない（部分観測）。

    歩容の位相 sin/cos（2）、関節角 - HOME（12）、roll・pitch・yaw（3）。
    """
    joint_angles = state.qpos[:, 7:] - home()
    n_env = joint_angles.shape[0]

    angle = 2 * np.pi * FREQ * t
    phase = torch.tensor(
        [np.sin(angle), np.cos(angle)],
        dtype=joint_angles.dtype,
        device=joint_angles.device,
    ).expand(n_env, 2)

    attitude = torch.stack(rpy(state.qpos[:, 3:7]), dim=1)
    return torch.cat([phase, joint_angles, attitude], dim=1)


class Net(nn.Module):
    """12 関節の目標角（HOME からの差）を直接出力するネットワーク。

    kind="gru": GRU（隠れ 32）。kind="mlp": 記憶を持たない比較用の MLP。
    """

    def __init__(self, kind="gru", hidden=32):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.core = nn.GRU(17, hidden, batch_first=True)
        else:
            self.core = nn.Sequential(nn.Linear(17, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 12)

        # 出力層は 0 で初期化する（最初は目標角 = HOME）
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs, h):
        """1 ステップ分。obs: (B, 17)。(目標角のずれ, 次の隠れ状態) を返す。"""
        if self.kind == "gru":
            y, h = self.core(obs[:, None], h)
            return torch.tanh(self.head(y[:, 0])), h
        return torch.tanh(self.head(self.core(obs))), h

    def sequence(self, obs):
        """系列全体を一度に処理する（模倣学習用）。obs: (B, T, 17)。"""
        if self.kind == "gru":
            y, _ = self.core(obs)
            return torch.tanh(self.head(y))
        return torch.tanh(self.head(self.core(obs)))

    def init_h(self, n_env):
        if self.kind != "gru":
            return None
        return torch.zeros(1, n_env, self.head.in_features, device=gs.device)


class OpenLoop(nn.Module):
    """ベースライン: 歩容だけ（フィードバックなし）。evaluate に net として渡せる。"""

    def init_h(self, n_env):
        self.t = 0.0
        return None

    def forward(self, obs, h):
        offset = gait_offset(self.t).expand(obs.shape[0], 12)
        self.t += DT
        return offset, h


class RobotState:
    """複数台のロボットを 1 つのシーンに置いたとき、k 台目の状態だけを切り出す。"""

    def __init__(self, state, k):
        self.qpos = state.qpos[:, 19 * k:19 * (k + 1)]
        self.dofs_vel = state.dofs_vel[:, 18 * k:18 * (k + 1)]


def reset_episode(scene, robot, n_env, gen, yaw_range=0.6):
    """初期ヨーを ±yaw_range の一様乱数で決めて、立位に戻す。"""
    scene.reset()

    yaw0 = (torch.rand(n_env, generator=gen) * 2 - 1) * yaw_range
    quat = torch.stack(
        [torch.cos(yaw0 / 2), torch.zeros(n_env), torch.zeros(n_env), torch.sin(yaw0 / 2)],
        dim=1,
    )
    robot.set_quat(quat.to(gs.tc_float).to(gs.device))
    robot.set_dofs_position(home().expand(n_env, 12).contiguous(), DOFS)


def apply_target(robot, state, target):
    """目標角に向けた PD トルクを与える（勾配が流れるよう control_dofs_force を使う）。"""
    angle = state.qpos[:, 7:]
    velocity = state.dofs_vel[:, 6:]
    torque = KP * (target - angle) - KV * velocity
    robot.control_dofs_force(torque, DOFS)


def step_policy(scene, robot, net, h, t):
    """ネットワークで 1 ステップ進める。(次の状態, 出力, 次の隠れ状態) を返す。"""
    state = scene.sim.rigid_solver.get_state()  # 勾配つきで読めるのはこの状態
    action, h = net(observe(state, t), h)
    apply_target(robot, state, home() + action)
    scene.step()
    next_state = scene.sim.rigid_solver.get_state()
    return next_state, action, h


# ---- 足先の運動学（滑りの計算用）: Go2 の寸法 ----
# 股関節の位置 (x, y)。順序は FL, FR, RL, RR
LEG_XY = torch.tensor([[0.1934, 0.0465], [0.1934, -0.0465], [-0.1934, 0.0465], [-0.1934, -0.0465]])
LEG_Y = torch.tensor([0.0955, -0.0955, 0.0955, -0.0955])  # thigh の横オフセット
LEG_L = 0.213                                             # thigh・calf それぞれの長さ


def quat_to_R(quat):
    """クォータニオン (B, 4) を回転行列 (B, 3, 3) にする。"""
    w, x, y, z = quat.unbind(-1)
    rows = [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ]
    return torch.stack(rows, dim=-1).reshape(-1, 3, 3)


def foot_pos(state):
    """足先の世界座標 (B, 4, 3)。関節角からの順運動学で求める（微分可能）。"""
    q = state.qpos[:, 7:]
    hip, thigh, calf = q[:, 0::3], q[:, 1::3], q[:, 2::3]
    hip_xy = LEG_XY.to(q)
    side_offset = LEG_Y.to(q)

    # 脚の面内（前後 x, 上下 z）の足先位置
    x = -LEG_L * torch.sin(thigh) - LEG_L * torch.sin(thigh + calf)
    z = -LEG_L * torch.cos(thigh) - LEG_L * torch.cos(thigh + calf)

    # hip（外転）の回転を反映して横 y と高さを求める
    y_body = torch.cos(hip) * side_offset - torch.sin(hip) * z
    z_body = torch.sin(hip) * side_offset + torch.cos(hip) * z

    foot_in_body = torch.stack([x + hip_xy[:, 0], y_body + hip_xy[:, 1], z_body], dim=-1)
    rotation = quat_to_R(state.qpos[:, 3:7])
    foot_in_world = torch.einsum("bij,bkj->bki", rotation, foot_in_body)
    return state.qpos[:, None, 0:3] + foot_in_world


def contact_slip(foot, foot_prev):
    """足ごとの接地度 contact と、足先の水平速度の二乗 slip2 を返す。

    接地度は足先の高さからの滑らかな判定（高さ 2.6 cm 付近で 0.5）。
    """
    velocity = (foot - foot_prev) / DT
    contact = torch.sigmoid((0.026 - foot[..., 2]) / 0.003)
    slip2 = velocity[..., 0] ** 2 + velocity[..., 1] ** 2
    return contact, slip2


# ---- 損失 ----
LAM = 0.98                  # 脚の動きの大きさの移動平均の係数（約 1 歩容周期）
PAIRS = [(0, 1), (2, 3)]    # 左右ペア (FL, FR), (RL, RR)

# 勾配学習の重み
LOSS_WEIGHT = {
    "vx": 30.0,    # 前進速度の誤差
    "dq": 0.003,   # 関節速度（動きすぎ）
    "sym": 2.0,    # 左右非対称
    "ang": 0.1,    # 胴体の角速度（頭の振れ）
    "slip": 0.5,   # 足の滑り
}


def sym_pen(energy):
    """左右の脚の動きの大きさの相対差の二乗和（thigh と calf）。

    energy は (q - HOME)^2 の移動平均で、形は (B, 12)。
    """
    penalty = 0.0
    for left, right in PAIRS:
        for joint in (1, 2):
            e_left = energy[:, 3 * left + joint]
            e_right = energy[:, 3 * right + joint]
            penalty = penalty + ((e_left - e_right) / (e_left + e_right + 1e-4)) ** 2
    return penalty


def step_loss(state):
    """1 ステップ分の損失（環境ごと、形は (B,)）。滑りと左右対称は呼び出し側で足す。"""
    roll, pitch, yaw = rpy(state.qpos[:, 3:7])
    height = state.qpos[:, 2]
    vx = state.dofs_vel[:, 0]
    vy = state.dofs_vel[:, 1]
    joint_speed2 = (state.dofs_vel[:, 6:] ** 2).sum(dim=1)
    body_rate2 = (state.dofs_vel[:, 3:6] ** 2).sum(dim=1)

    loss = 10 * yaw ** 2                                     # ヨーを 0 に戻す
    loss = loss + LOSS_WEIGHT["vx"] * (vx - V_TGT) ** 2      # 前進速度を目標に
    loss = loss + 0.5 * vy ** 2                              # 横に流れない
    loss = loss + 3 * (roll ** 2 + pitch ** 2)               # 傾かない
    loss = loss + 20 * torch.relu(0.25 - height) ** 2        # 沈み込まない
    loss = loss + LOSS_WEIGHT["dq"] * joint_speed2
    loss = loss + LOSS_WEIGHT["ang"] * body_rate2
    return loss


# ---- 評価 ----
@torch.no_grad()
def evaluate(scene, robot, net, n_env, seed=1234, record=False):
    """初期ヨーを seed で決めた n_env 体を 6 秒歩かせて、指標の辞書を返す。

    record=True なら、0 番のロボットの軌跡（qpos, dofs_vel, 出力）も返す。
    """
    gen = torch.Generator().manual_seed(seed)
    reset_episode(scene, robot, n_env, gen)
    h = net.init_h(n_env)

    fell = torch.zeros(n_env, dtype=torch.bool, device=gs.device)
    yaws, vxs, joint_speeds2, body_rates2, trajectory = [], [], [], [], []
    leg_energy = torch.zeros(n_env, 12, device=gs.device)  # (関節角 - HOME)^2 の合計（左右対称の指標用）
    slip_sum, contact_sum, foot_prev = 0.0, 0.0, None

    for i in range(T_EP):
        state, action, h = step_policy(scene, robot, net, h, i * DT)

        fell |= state.qpos[:, 2] < 0.18
        yaws.append(rpy(state.qpos[:, 3:7])[2])
        vxs.append(state.dofs_vel[:, 0])
        joint_speeds2.append((state.dofs_vel[:, 6:] ** 2).sum(dim=1))
        body_rates2.append((state.dofs_vel[:, 3:6] ** 2).sum(dim=1))

        foot = foot_pos(state)
        if i >= 100:  # 歩き出しの最初の 1 秒は除く
            leg_energy += (state.qpos[:, 7:] - home()) ** 2
            contact, slip2 = contact_slip(foot, foot_prev)
            slip_sum = slip_sum + (contact * slip2.sqrt()).sum()
            contact_sum = contact_sum + contact.sum()
        foot_prev = foot

        if record:
            row = torch.cat([state.qpos, state.dofs_vel, action], dim=1)[0]
            trajectory.append(row.cpu().numpy())

    yaw = torch.stack(yaws)
    vx = torch.stack(vxs)
    leg_rms = (leg_energy / (T_EP - 100)).sqrt()

    # 左右非対称: 左右の脚の動きの大きさの相対差（thigh と calf）
    asymmetry = []
    for left, right in PAIRS:
        for joint in (1, 2):
            r_left = leg_rms[:, 3 * left + joint]
            r_right = leg_rms[:, 3 * right + joint]
            asymmetry.append((r_left - r_right).abs() / (r_left + r_right + 1e-6))
    asymmetry = torch.stack(asymmetry).mean()

    metrics = {
        "fall_rate": fell.float().mean().item(),
        "yaw_abs": yaw.abs().mean().item(),
        "yaw_abs_last2s": yaw[-200:].abs().mean().item(),
        "vx_mean": vx[100:].mean().item(),
        "vx_err": (vx[100:] - V_TGT).abs().mean().item(),
        "y_abs_end": state.qpos[:, 1].abs().mean().item(),
        "dq_rms": torch.stack(joint_speeds2).mean().sqrt().item() / 12 ** 0.5,
        "ang_rms": torch.stack(body_rates2).mean().sqrt().item(),
        "slip": (slip_sum / (contact_sum + 1e-6)).item(),
        "sym_asym": asymmetry.item(),
    }
    if record:
        return metrics, np.array(trajectory)
    return metrics


def score(metrics):
    """モデル選択用の総合スコア（小さいほど良い）。"""
    return (
        metrics["yaw_abs"]
        + 3 * metrics["vx_err"]
        + 2 * metrics["fall_rate"]
        + 0.05 * metrics["dq_rms"]
        + 0.5 * metrics["sym_asym"]
        + 0.1 * metrics["ang_rms"]
        + 0.5 * metrics["slip"]
    )


def evaluate_all(scene, robot, net, n_env, seeds=(777, 2024, 4242)):
    """複数の初期条件で評価して、指標ごとに平均する（報告用）。"""
    results = [evaluate(scene, robot, net, n_env, seed=s) for s in seeds]
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}
