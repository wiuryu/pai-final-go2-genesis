"""動画を作る。

  compare: 開ループ歩容と学習済み GRU を並べて走らせ、ヨー・横位置・前進距離を比べる
  inside : GRU の入力・隠れ状態・出力を、歩行の映像と一緒に可視化する
  summary: 提出用のまとめ動画。タイトル画面つきで、学習の 3 段階の比較、学習前と学習後の比較、
           経路上の点を順に通過するタスクをつなげる

使い方: python video.py compare|inside|summary --ckpt models/gru_final.pt [--out out.mp4]
summary は --ckpt1 で段階 1 の重み（既定: results/imitate_final.pt）も使う。
（動画の書き出しに ffmpeg が必要）
"""
import argparse
import subprocess

import numpy as np
import torch
import genesis as gs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import walker as W
import waypoints as WP
from imitate import expert_target

matplotlib.rcParams["font.family"] = ["Hiragino Sans", "IPAexGothic", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

VIDEO_W, VIDEO_H, FPS = 1920, 1080, 50
FRAME_EVERY = 2  # 2 ステップ（20 ms）ごとに 1 フレーム = 実時間の等倍
JOINT_NAMES = [f"{leg}_{joint}" for leg in ["FL", "FR", "RL", "RR"] for joint in ["hip", "thigh", "calf"]]
THIGH_INDEX = [1, 4, 7, 10]
LEG_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
OPEN_LOOP_COLOR, GRU_COLOR, NAVY = "#d95f02", "#1f77b4", "#1f3a6b"
HEADING_SIZE, SUBHEADING_SIZE = 28, 20     # 画面上部の見出しと、各映像の見出しの文字の大きさ
TITLE_SECONDS, HOLD_SECONDS = 5, 3          # タイトル画面の長さと、映像を流す前に止める長さ [s]


def make_scene(n_robots, resolutions, spacing=2.0, stagger=0.0, markers=()):
    """ロボットを n_robots 台、y 方向に spacing [m]、x 方向に stagger [m] ずつずらして置く。

    resolutions の数だけカメラを作る。x 方向にずらすと、横から映したときに隣のロボットが視界から外れる。
    markers は、赤い柱を立てる位置 (x, y) のリスト（経路上の点の目印）。
    """
    gs.init(backend=gs.cpu, logging_level="error")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=W.DT, requires_grad=True),
        vis_options=gs.options.VisOptions(show_world_frame=False),
    )
    scene.add_entity(gs.morphs.Plane())

    robots = []
    for i in range(n_robots):
        robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(stagger * i, spacing * i, 0.42)))
        robots.append(robot)

    for x, y in markers:
        pole = gs.morphs.Cylinder(radius=0.03, height=0.6, pos=(x, y, 0.3), fixed=True, collision=False)
        scene.add_entity(pole, surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0)))

    cameras = []
    for resolution in resolutions:
        camera = scene.add_camera(res=resolution, pos=(1, -3, 1.2), lookat=(0, 1, 0.3), fov=45, GUI=False)
        cameras.append(camera)

    scene.build(n_envs=1)
    return scene, robots, cameras


def load_net(ckpt):
    """学習済み GRU を読む。gs.init の後（デバイスが決まってから）に呼ぶ。"""
    net = W.Net().to(gs.device)
    net.load_state_dict(torch.load(ckpt))
    return net.eval()


@torch.no_grad()
def rollout(scene, robots, nets, yaw0, capture):
    """全ロボットを 6 秒歩かせて、ステップごとのログと映像のフレームを返す。

    nets[k] が None のロボットは開ループ（歩容のみ）、そうでなければそのネットワークで歩く。
    capture(全ロボットの位置) は、その時刻の映像（フレーム）を返す関数。
    """
    scene.reset()
    quat = torch.tensor(
        [[np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]],
        dtype=gs.tc_float,
        device=gs.device,
    )
    for robot in robots:
        robot.set_quat(quat)
        robot.set_dofs_position(W.home().expand(1, 12).contiguous(), W.DOFS)

    hidden = [net.init_h(1) if net is not None else None for net in nets]
    log = {key: [] for key in ["pos", "yaw", "obs", "hidden", "action", "expert_action"]}
    frames = []

    for i in range(W.T_EP):
        t = i * W.DT
        state = scene.sim.rigid_solver.get_state()
        step = {key: [] for key in log}

        for k, robot in enumerate(robots):
            robot_state = W.RobotState(state, k)

            if nets[k] is None:
                action = W.gait_offset(t)[None]
                step["obs"].append(None)
                step["hidden"].append(None)
                step["expert_action"].append(None)
            else:
                obs = W.observe(robot_state, t)
                action, hidden[k] = nets[k](obs, hidden[k])
                step["obs"].append(obs[0].cpu().numpy())
                step["hidden"].append(hidden[k][0, 0].cpu().numpy())
                step["expert_action"].append(expert_target(robot_state, t)[0].cpu().numpy())

            W.apply_target(robot, robot_state, W.home() + action)
            step["pos"].append(robot_state.qpos[0, :3].cpu().numpy())
            step["yaw"].append(W.rpy(robot_state.qpos[:, 3:7])[2].item())
            step["action"].append(action[0].cpu().numpy())

        scene.step()
        for key in log:
            log[key].append(step[key])

        if i % FRAME_EVERY == 0:
            frames.append(capture(np.array(step["pos"])))

    return log, frames


def start_encoder(out):
    """生のフレームを標準入力から受け取って mp4 にする ffmpeg を起動する。"""
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out,
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def write_frame(fig, encoder, repeat=1):
    """いまの図を repeat フレーム分書き出す。"""
    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba()).tobytes()
    for _ in range(repeat):
        encoder.stdin.write(data)


def hold_repeat(i, n_frames):
    """最初と最後の画面は HOLD_SECONDS 秒止める。"""
    if i == 0 or i == n_frames - 1:
        return HOLD_SECONDS * FPS
    return 1


def frame_to_step(frame_index):
    return min(FRAME_EVERY * frame_index, W.T_EP - 1)


def side_camera(camera, center):
    """ロボットを横から映すカメラ。center は映したい位置（x, y, z）。"""
    camera.set_pose(pos=(center[0] + 0.8, center[1] - 4.3, 1.5), lookat=(center[0], center[1], 0.25))


BOTTOM_ROW = [[0.05 + j * 0.32, 0.07, 0.27, 0.22] for j in range(3)]  # 下段に横に 3 つ
RIGHT_COLUMN = [[0.56, 0.66, 0.41, 0.20], [0.56, 0.37, 0.41, 0.20], [0.56, 0.08, 0.41, 0.20]]  # 右側に縦に 3 つ


def add_graph_panels(fig, t, panels, rects=BOTTOM_ROW):
    """3 つのグラフを rects の位置に並べ、現在時刻を示す縦線のリストを返す。

    panels は (タイトル, 曲線のリスト, 縦軸の範囲) のリスト。
    曲線は (ラベル, 色, 値の系列) で、縦軸の範囲が None なら自動で決める。
    """
    time_lines = []
    for j, ((title, curves, y_limit), rect) in enumerate(zip(panels, rects)):
        ax = fig.add_axes(rect)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("時間 [s]", fontsize=10)
        ax.set_xlim(0, t[-1])
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3)
        for label, color, values in curves:
            ax.plot(t, values, color=color, lw=1.8, label=label)
        if j == 0:
            ax.legend(fontsize=10)
            ax.axhline(0, color="k", lw=0.6)
        time_lines.append(ax.axvline(0, color="k", lw=1.2))
    return time_lines


def add_video_panel(fig, first_frame, title, rect=(0.02, 0.36, 0.96, 0.60)):
    """映像を rect の位置に置く。映像と、時刻を書き込む文字を返す。"""
    ax = fig.add_axes(rect)
    ax.axis("off")
    image = ax.imshow(first_frame)
    ax.set_title(title, fontsize=SUBHEADING_SIZE, loc="left")
    caption = ax.text(
        0.99, 0.03, "", transform=ax.transAxes, ha="right", va="bottom",
        color="w", fontsize=14, bbox=dict(fc="k", alpha=0.5, lw=0),
    )
    return image, caption


def compare(ckpt, out):
    scene, robots, cameras = make_scene(n_robots=2, resolutions=[(1280, 720)])
    camera = cameras[0]
    net = load_net(ckpt)
    encoder = start_encoder(out)

    def capture(positions):
        side_camera(camera, positions.mean(axis=0))
        return camera.render()[0]

    for yaw0 in (0.5, -0.5):
        log, frames = rollout(scene, robots, [None, net], yaw0, capture)
        t = np.arange(W.T_EP) * W.DT
        yaw_deg = np.degrees(np.array(log["yaw"]))  # (T, 2): 開ループ, GRU
        pos = np.array(log["pos"])                  # (T, 2, 3)
        y_from_start = pos[:, :, 1] - pos[0, :, 1]
        x_from_start = pos[:, :, 0] - pos[0, :, 0]

        fig = plt.figure(figsize=(VIDEO_W / 100, VIDEO_H / 100), dpi=100)
        title = (
            f"同じ初期ヨー（{np.degrees(yaw0):+.0f}°）から出発　"
            "手前: 歩容のみ（開ループ） / 奥: GRU（部分観測・学習済み）"
        )
        image, caption = add_video_panel(fig, frames[0], title)

        def both(values):
            return [("開ループ", OPEN_LOOP_COLOR, values[:, 0]), ("GRU", GRU_COLOR, values[:, 1])]

        panels = [
            ("ヨー（進行方向）[deg]", both(yaw_deg), None),
            ("横方向の位置 y（出発点から）[m]", both(y_from_start), None),
            ("前進距離 x [m]", both(x_from_start), None),
        ]
        time_lines = add_graph_panels(fig, t, panels)

        for i, frame in enumerate(frames):
            step = frame_to_step(i)
            image.set_data(frame)
            caption.set_text(
                f"t = {t[step]:4.2f} s　ヨー: 開ループ {yaw_deg[step, 0]:+5.1f}°  GRU {yaw_deg[step, 1]:+5.1f}°"
            )
            for line in time_lines:
                line.set_xdata([t[step], t[step]])
            write_frame(fig, encoder)
        plt.close(fig)

    encoder.stdin.close()
    encoder.wait()
    print("saved", out)


def inside(ckpt, out, yaw0=0.5):
    def move_camera(camera, positions):
        robot_pos = positions[0]
        camera.set_pose(pos=(robot_pos[0] + 1.3, robot_pos[1] - 1.1, 0.6), lookat=(robot_pos[0], robot_pos[1], 0.25))

    scene, robots, cameras = make_scene(n_robots=1, resolutions=[(960, 540)])
    camera = cameras[0]
    net = load_net(ckpt)

    def capture(positions):
        move_camera(camera, positions)
        return camera.render()[0]

    log, frames = rollout(scene, robots, [net], yaw0, capture)

    t = np.arange(W.T_EP) * W.DT
    obs = np.array([row[0] for row in log["obs"]])
    hidden = np.array([row[0] for row in log["hidden"]])
    action = np.array([row[0] for row in log["action"]])
    expert = np.array([row[0] for row in log["expert_action"]])
    yaw_deg = np.degrees(np.array(log["yaw"])[:, 0])

    def standardize(matrix):
        return (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-6)

    fig = plt.figure(figsize=(VIDEO_W / 100, VIDEO_H / 100), dpi=100)
    time_lines = []

    # 左上: 映像
    video_ax = fig.add_axes([0.02, 0.45, 0.46, 0.50])
    video_ax.axis("off")
    image = video_ax.imshow(frames[0])
    video_ax.set_title(f"学習済み GRU（部分観測）で歩行　初期ヨー {np.degrees(yaw0):+.0f}°", fontsize=14, loc="left")
    caption = video_ax.text(
        0.99, 0.03, "", transform=video_ax.transAxes, ha="right", va="bottom",
        color="w", fontsize=13, bbox=dict(fc="k", alpha=0.5, lw=0),
    )

    def add_heatmap(rect, matrix, row_labels, title, label_size):
        """行 = 次元、列 = 時間のヒートマップ。赤が正、青が負。"""
        ax = fig.add_axes(rect)
        limit = np.abs(matrix).max()
        ax.imshow(
            matrix.T, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
            extent=[0, t[-1], matrix.shape[1], 0], interpolation="nearest",
        )
        ax.set_yticks(np.arange(matrix.shape[1]) + 0.5)
        ax.set_yticklabels(row_labels, fontsize=label_size)
        ax.set_title(title, fontsize=11, loc="left")
        ax.tick_params(axis="x", labelsize=8)
        time_lines.append(ax.axvline(0, color="k", lw=1.4))

    # 右: 入力 → 隠れ状態 → 出力
    input_labels = ["sin(位相)", "cos(位相)"] + JOINT_NAMES + ["roll", "pitch", "yaw"]
    add_heatmap([0.55, 0.64, 0.43, 0.31], standardize(obs), input_labels,
                "① 入力（17 次元：位相・関節角・姿勢。速度は入っていない。各行を標準化）", 7)
    add_heatmap([0.55, 0.37, 0.43, 0.22], standardize(hidden), [f"h{i}" for i in range(32)],
                "② GRU の隠れ状態（32 次元。各行を標準化）", 5)
    add_heatmap([0.55, 0.07, 0.43, 0.25], action, JOINT_NAMES,
                "③ 出力＝12 関節の目標角（HOME からの差）[rad]", 7)

    # 左中: ヨー
    yaw_ax = fig.add_axes([0.05, 0.28, 0.43, 0.11])
    yaw_ax.plot(t, yaw_deg, color=GRU_COLOR, lw=1.6)
    yaw_ax.axhline(0, color="k", lw=0.6)
    yaw_ax.grid(alpha=0.3)
    yaw_ax.set_xlim(0, t[-1])
    yaw_ax.set_title("ヨー [deg]（0 に戻ると直進）", fontsize=11, loc="left")
    yaw_ax.tick_params(labelsize=8)
    time_lines.append(yaw_ax.axvline(0, color="k", lw=1.2))

    # 左下: thigh の目標角を、模倣元の教師と重ねる
    thigh_ax = fig.add_axes([0.05, 0.07, 0.43, 0.16])
    for k, joint in enumerate(THIGH_INDEX):
        leg = ["FL", "FR", "RL", "RR"][k]
        thigh_ax.plot(t, action[:, joint], color=LEG_COLORS[k], lw=1.4, label=leg)
        thigh_ax.plot(t, expert[:, joint], color=LEG_COLORS[k], lw=1, ls=":")
    thigh_ax.set_title("thigh の目標角（HOME からの差）：GRU（実線）と模倣元の教師（点線）[rad]", fontsize=11, loc="left")
    thigh_ax.set_xlabel("時間 [s]", fontsize=9)
    thigh_ax.set_xlim(0, t[-1])
    thigh_ax.grid(alpha=0.3)
    thigh_ax.tick_params(labelsize=8)
    thigh_ax.legend(fontsize=8, ncol=4, loc="upper right")
    time_lines.append(thigh_ax.axvline(0, color="k", lw=1.2))

    encoder = start_encoder(out)
    for i, frame in enumerate(frames):
        step = frame_to_step(i)
        image.set_data(frame)
        caption.set_text(f"t = {t[step]:4.2f} s　ヨー {yaw_deg[step]:+5.1f}°")
        for line in time_lines:
            line.set_xdata([t[step], t[step]])
        write_frame(fig, encoder)
    encoder.stdin.close()
    encoder.wait()

    correlation = np.corrcoef(action[:, THIGH_INDEX].ravel(), expert[:, THIGH_INDEX].ravel())[0, 1]
    print("saved", out, "| thigh 目標の GRU と教師の相関:", round(float(correlation), 3))


STAGE_TITLES = ["手書き歩容（開ループ）", "学習段階 1（模倣のみ）", "学習段階 2（最終）"]
SPACING, STAGGER = 6.0, 10.0   # ロボットを置く間隔。ずらして置くと、横から映したとき隣が写り込まない


def oblique_camera(camera, position, distance=1.6, height=0.9):
    """ロボットを斜め上から映す。ロボットの前進に合わせて動く。"""
    x, y = position[0], position[1]
    camera.set_pose(pos=(x + 0.6, y - distance, height), lookat=(x, y, 0.25))


def render_view(camera, position, crop_width=None):
    """斜め上のカメラで映した画像を返す。crop_width を指定すると、中央をその幅で切り取る（縦長にする）。"""
    oblique_camera(camera, position)
    image = camera.render()[0]
    if crop_width is None:
        return image
    left = (image.shape[1] - crop_width) // 2
    return image[:, left:left + crop_width]


def chase_camera(camera, position):
    """ロボットの後ろの斜め上から、進行方向（+x）を映す。経路上の点が見える。"""
    x, y = position[0], position[1]
    camera.set_pose(pos=(x - 2.4, y - 0.5, 1.1), lookat=(x + 1.5, y, 0.3))


def add_heading(fig, text):
    """画面の上部の見出し。全ての画面で、位置と大きさを同じにする。"""
    fig.text(0.02, 0.985, text, fontsize=HEADING_SIZE, va="top")


def write_title_card(encoder, title, subtitle, seconds=TITLE_SECONDS):
    """映像の前に、タイトル画面を seconds 秒表示する。"""
    fig = plt.figure(figsize=(VIDEO_W / 100, VIDEO_H / 100), dpi=100)
    fig.text(0.5, 0.58, title, ha="center", va="center", fontsize=34, color=NAVY)
    fig.text(0.5, 0.42, subtitle, ha="center", va="center", fontsize=20, linespacing=1.8)
    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba()).tobytes()
    for _ in range(int(seconds * FPS)):
        encoder.stdin.write(data)
    plt.close(fig)


def write_three_tiles(encoder, view_frames, yaw_deg, heading):
    """3 台を縦長の映像で左右に並べる。各映像の下に時刻とヨーを表示する。最初の画面は止めてから流す。"""
    rects = [[0.02, 0.12, 0.30, 0.70], [0.35, 0.12, 0.30, 0.70], [0.68, 0.12, 0.30, 0.70]]
    n_steps = yaw_deg.shape[0]
    fig = plt.figure(figsize=(VIDEO_W / 100, VIDEO_H / 100), dpi=100)
    add_heading(fig, heading)

    images, captions = [], []
    for frames, rect, title in zip(view_frames, rects, STAGE_TITLES):
        ax = fig.add_axes(rect)
        ax.axis("off")
        images.append(ax.imshow(frames[0]))
        ax.set_title(title, fontsize=SUBHEADING_SIZE)
        captions.append(fig.text(rect[0] + rect[2] / 2, rect[1] - 0.045, "", ha="center", fontsize=15))

    n_frames = len(view_frames[0])
    for i in range(n_frames):
        step = min(FRAME_EVERY * i, n_steps - 1)
        for v, (image, caption) in enumerate(zip(images, captions)):
            image.set_data(view_frames[v][i])
            caption.set_text(f"t = {step * W.DT:4.2f} s　ヨー {yaw_deg[step, v]:+5.1f}°")
        write_frame(fig, encoder, repeat=hold_repeat(i, n_frames))
    plt.close(fig)


def write_before_after(encoder, frames_before, frames_after, yaw_deg, pos, yaw0):
    """左に学習前（上）と学習後（下）の映像、右にヨー・横位置・前進距離のグラフを並べる。"""
    t = np.arange(W.T_EP) * W.DT
    fig = plt.figure(figsize=(VIDEO_W / 100, VIDEO_H / 100), dpi=100)
    add_heading(fig, f"学習前と学習後の比較　初期ヨー {np.degrees(yaw0):+.0f}°")

    top = add_video_panel(fig, frames_before[0], "学習前（手書き歩容・開ループ）", rect=(0.02, 0.47, 0.46, 0.38))
    bottom = add_video_panel(fig, frames_after[0], "学習後（学習段階 2・最終）", rect=(0.02, 0.03, 0.46, 0.38))

    y_from_start = pos[:, :, 1] - pos[0, :, 1]
    x_from_start = pos[:, :, 0] - pos[0, :, 0]

    def both(values):
        return [("学習前", OPEN_LOOP_COLOR, values[:, 0]), ("学習後", GRU_COLOR, values[:, 1])]

    panels = [
        ("ヨー（進行方向）[deg]", both(yaw_deg), None),
        ("横方向の位置 y（出発点から）[m]", both(y_from_start), None),
        ("前進距離 x [m]", both(x_from_start), None),
    ]
    time_lines = add_graph_panels(fig, t, panels, rects=RIGHT_COLUMN)

    n_frames = len(frames_before)
    for i in range(n_frames):
        step = frame_to_step(i)
        for (image, caption), frame, column in ((top, frames_before[i], 0), (bottom, frames_after[i], 1)):
            image.set_data(frame)
            caption.set_text(f"t = {t[step]:4.2f} s　ヨー {yaw_deg[step, column]:+5.1f}°")
        for line in time_lines:
            line.set_xdata([t[step], t[step]])
        write_frame(fig, encoder, repeat=hold_repeat(i, n_frames))
    plt.close(fig)


def write_waypoint_task(encoder, frames, record, targets, passed):
    """経路上の点を通る映像を書き出す。上に映像、下に真上から見た地図（点・軌跡・現在位置）を置く。"""
    n_frames = len(frames)
    t_end = len(record["pos"]) * W.DT
    fig = plt.figure(figsize=(VIDEO_W / 100, VIDEO_H / 100), dpi=100)
    add_heading(fig, "経路上の点を順に通過する（学習段階 2・再学習なし）")
    image, caption = add_video_panel(fig, frames[0], "", rect=(0.20, 0.30, 0.60, 0.60))

    ax = fig.add_axes([0.05, 0.05, 0.90, 0.20])
    ax.set_aspect("equal")
    ax.set_title("真上から見た経路（円: 通過とみなす範囲、緑: 通過済み）", fontsize=12, loc="left")
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.3)
    circles = []
    for number, (x, y) in enumerate(targets, start=1):
        circle = plt.Circle((x, y), WP.REACH_RADIUS, fc="#f4b6b6", ec="#c0392b", lw=1.5)
        ax.add_patch(circle)
        ax.text(x, y, str(number), ha="center", va="center", fontsize=10)
        circles.append(circle)
    xs = [x for x, _ in targets] + [record["pos"][0][0]]
    ys = [y for _, y in targets]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
    (track_line,) = ax.plot([], [], color=GRU_COLOR, lw=2)
    (robot_dot,) = ax.plot([], [], "o", color=GRU_COLOR, ms=9)

    for i in range(n_frames):
        step = min(FRAME_EVERY * i, len(record["pos"]) - 1)
        t = step * W.DT
        n_passed = sum(p is not None and p <= t for p in passed)
        for circle, p in zip(circles, passed):
            if p is not None and p <= t:
                circle.set_facecolor("#9bd6a8")
                circle.set_edgecolor("#2e8b57")
        track = np.array(record["pos"][: step + 1])
        track_line.set_data(track[:, 0], track[:, 1])
        robot_dot.set_data([track[-1, 0]], [track[-1, 1]])
        image.set_data(frames[i])
        caption.set_text(f"t = {t:4.1f} s　通過した点 {n_passed} / {len(targets)}")
        write_frame(fig, encoder, repeat=hold_repeat(i, n_frames))
    plt.close(fig)


def summary(ckpt_stage1, ckpt_final, out):
    """提出用のまとめ動画。各映像の前にタイトル画面を 3 秒表示する。"""
    tile = (960, 720)   # 横長で映してから、中央を縦長に切り取る
    tile_crop = 540
    split = (768, 432)
    task = (1152, 648)
    robot_origin = (STAGGER * 3, SPACING * 3)  # 4 台目の出発点。経路の点はここからの相対位置（他の映像に写らない離れた場所）
    targets = [(robot_origin[0] + x, robot_origin[1] + y) for x, y in WP.WAYPOINTS]
    scene, robots, cameras = make_scene(
        n_robots=4,
        resolutions=[tile, tile, tile, split, split, task],
        spacing=SPACING,
        stagger=STAGGER,
        markers=targets,
    )
    tile_cameras, split_cameras, task_camera = cameras[:3], cameras[3:5], cameras[5]
    nets = [None, load_net(ckpt_stage1), load_net(ckpt_final)]
    encoder = start_encoder(out)

    def capture_tiles(positions):
        frames = []
        for camera, position in zip(tile_cameras, positions):
            frames.append(render_view(camera, position, crop_width=tile_crop))
        return frames

    def capture_split(positions):
        frames = []
        for camera, position in zip(split_cameras, (positions[0], positions[2])):
            frames.append(render_view(camera, position))
        return frames

    write_title_card(
        encoder,
        "四脚ロボット Go2 を GRU と微分可能シミュレータで歩かせる",
        "シミュレータ: Genesis　　すべて等倍の再生（倍速なし）",
    )

    # 1. 学習の 3 段階を並べて比べる
    for yaw0 in (0.5, -0.5):
        write_title_card(
            encoder,
            "学習の 3 段階を並べて比べる",
            f"初期ヨー {np.degrees(yaw0):+.0f}°\n左から　手書き歩容（開ループ）　学習段階 1（模倣のみ）　学習段階 2（最終）",
        )
        log, frames = rollout(scene, robots[:3], nets, yaw0, capture_tiles)
        views = [[frame[v] for frame in frames] for v in range(3)]
        yaw_deg = np.degrees(np.array(log["yaw"]))
        heading = f"同じ初期ヨー（{np.degrees(yaw0):+.0f}°）から出発した 3 台"
        write_three_tiles(encoder, views, yaw_deg, heading)

    # 2. 学習前と学習後を、グラフと一緒に比べる
    for yaw0 in (0.5, -0.5):
        write_title_card(
            encoder,
            "学習前と学習後の比較",
            f"初期ヨー {np.degrees(yaw0):+.0f}°\n左　上: 学習前（手書き歩容）　下: 学習後（最終）　　右: ヨー・横位置・前進距離",
        )
        log, frames = rollout(scene, robots[:3], nets, yaw0, capture_split)
        yaw_deg = np.degrees(np.array(log["yaw"]))[:, [0, 2]]
        pos = np.array(log["pos"])[:, [0, 2], :]
        write_before_after(
            encoder, [f[0] for f in frames], [f[1] for f in frames], yaw_deg, pos, yaw0
        )

    # 3. 経路上の点を順に通過する（再学習なし）
    write_title_card(
        encoder,
        "経路上の点を順に通過する",
        "学習済みの GRU をそのまま使う（再学習なし）\n"
        "入力のヨーを「目標方向とのずれ」に置き換えて、次の点の方向へ向かわせる",
    )
    frames, record = [], {"pos": []}

    def on_step(t, state, next_index, heading):
        position = (state.qpos[0, 0].item(), state.qpos[0, 1].item())
        record["pos"].append(position)
        if round(t / W.DT) % FRAME_EVERY == 0:
            chase_camera(task_camera, position)
            frames.append(task_camera.render()[0])

    passed, track = WP.follow_waypoints(scene, robots[3], nets[2], robot_origin, WP.WAYPOINTS, on_step, index=3)
    write_waypoint_task(encoder, frames, record, targets, passed)

    encoder.stdin.close()
    encoder.wait()
    print("saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["compare", "inside", "summary"])
    parser.add_argument("--ckpt", default="models/gru_final.pt")
    parser.add_argument("--ckpt1", default="results/imitate_final.pt", help="summary 用: 学習段階 1 の重み")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or f"gru_{args.kind}.mp4"
    if args.kind == "compare":
        compare(args.ckpt, out)
    elif args.kind == "inside":
        inside(args.ckpt, out)
    else:
        summary(args.ckpt1, args.ckpt, out)


if __name__ == "__main__":
    main()
