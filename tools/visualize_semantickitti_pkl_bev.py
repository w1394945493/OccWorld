#!/usr/bin/env python3
"""可视化 SemanticKITTI OccWorld 风格 pkl 中的 Occupancy 与自车轨迹。

输入：
  1) semantickitti_converter.py 或后续完整 converter 保存的 pkl；
  2) 每帧 dense occupancy 标签文件。

输出：
  - 每个 scene 一个 BEV mp4；
  - 可选保存逐帧 png。

说明：
  - 本脚本面向 dense occupancy / SSC 标签，不适合直接可视化原始点云语义
  labels/*.label，除非该 .label 文件本身就是 H*W*D 展平后的体素标签。
  FoundationSSC 的确定性 dense occupancy 路径为：
    <ann_file>/<sequence>/<frame_id>_1_1.npy
  例如：
    /c20250502/.../dataset/labels/00/000000_1_1.npy
"""

import argparse
import os
import os.path as osp
import pickle
import sys
import time

import cv2
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/occworld_matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


"""SemanticKITTI class names:
'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
'pole', 'traffic-sign'
"""
SEMANTICKITTI_COLORS = np.array(
    [
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255],
    ]
).astype(np.float32) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SemanticKITTI OccWorld-style pkl as BEV videos.')
    parser.add_argument('--pkl', required=True, help='OccWorld-style pkl path')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--data-root', default='',
                        help='optional SemanticKITTI dataset root; empty means use metadata["data_root"] in pkl')
    parser.add_argument('--scene', default='',
                        help='scene key to visualize, e.g. sequence-00; empty means all scenes')
    parser.add_argument('--max-frames', type=int, default=-1,
                        help='max frames per scene; negative means all frames')
    parser.add_argument('--fps', type=float, default=5.0, help='output video FPS')
    parser.add_argument('--save-frames', action='store_true', help='also save png frames')
    parser.add_argument('--occ-shape', type=int, nargs=3, default=[256, 256, 32],
                        metavar=('H', 'W', 'D'),
                        help='kept for compatibility; FoundationSSC .npy uses saved shape')
    parser.add_argument('--occ-dtype', default='uint16',
                        choices=['uint8', 'uint16', 'uint32', 'int32', 'int64'],
                        help='dtype for raw dense occupancy labels')
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=[0.0, -25.6, -2.0, 51.2, 25.6, 4.4],
                        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                        help='voxel grid range in local LiDAR/ego coordinates')
    parser.add_argument('--empty-labels', type=int, nargs='*', default=[0, 255],
                        help='labels treated as empty/ignore in BEV projection')
    return parser.parse_args()


def progress_iter(iterable, total, desc='Progress'):
    """优先使用 tqdm；环境没有 tqdm 时退化为轻量级终端进度条。"""
    try:
        from tqdm import tqdm
        yield from tqdm(iterable, total=total, desc=desc)
        return
    except Exception:
        pass

    t_start = time.time()
    for idx, item in enumerate(iterable, 1):
        elapsed = time.time() - t_start
        fps = idx / max(elapsed, 1e-6)
        eta = (total - idx) / max(fps, 1e-6)
        print(f'\r{desc}: {idx}/{total} | {fps:.1f} frame/s | ETA {eta:.1f}s',
              end='', file=sys.stderr, flush=True)
        yield item
    print('', file=sys.stderr)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def quat_wxyz_to_rotmat(q):
    """wxyz 四元数转 3x3 旋转矩阵。"""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


def pose_matrix(info):
    """由 pkl 中 ego2global 平移和 wxyz 四元数构造 4x4 位姿。"""
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_wxyz_to_rotmat(info['ego2global_rotation'])
    mat[:3, 3] = np.asarray(info['ego2global_translation'], dtype=np.float64)
    return mat


def scene_trajectory_in_current(infos, current_idx):
    """把整个 scene 的自车位置统一变换到当前帧局部坐标系，用于叠加轨迹。"""
    poses = [pose_matrix(info) for info in infos]
    return scene_trajectory_from_pose_cache(poses, current_idx)


def scene_trajectory_from_pose_cache(poses, current_idx):
    """基于已缓存的4x4位姿，把整个scene轨迹变换到当前帧局部坐标系。"""
    global_to_current = np.linalg.inv(poses[current_idx])
    origin = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    traj = np.stack([(global_to_current @ pose @ origin)[:2] for pose in poses])
    return traj


def infer_data_root(data, args):
    """最多只需提供 data-root；未提供时从 pkl metadata 中读取。"""
    if args.data_root:
        return osp.abspath(args.data_root)
    metadata = data.get('metadata', {})
    data_root = metadata.get('data_root', '')
    if data_root:
        return osp.abspath(data_root)
    raise ValueError(
        'Cannot infer data_root. Please provide --data-root or use a pkl whose '
        'metadata contains "data_root".')


def resolve_occ_path(info, scene_name, data_root):
    """按 FoundationSSC 规则确定 dense occupancy 路径，不再枚举猜测。

    优先级：
      1) pkl frame_info 中的 occ_path；
      2) pkl frame_info 中的 voxel_path；
      3) <data_root>/labels/<sequence>/<token>_1_1.npy。
    """
    token = info['token']
    sequence = scene_name.replace('sequence-', '')
    for key in ['occ_path', 'voxel_path']:
        path = info.get(key, '')
        if path and osp.isfile(path):
            return path
    path = osp.join(data_root, 'labels', sequence, f'{token}_1_1.npy')
    if osp.isfile(path):
        return path
    raise FileNotFoundError(
        f'FoundationSSC occupancy not found: {path}. '
        'Expected deterministic path <data-root>/labels/<sequence>/<token>_1_1.npy.')


def load_occupancy(info, scene_name, data_root):
    """加载单帧 FoundationSSC dense occupancy：<seq>/<token>_1_1.npy。"""
    path = resolve_occ_path(info, scene_name, data_root)
    if path.endswith('.npy'):
        occ = np.load(path)
    elif path.endswith('.npz'):
        data = np.load(path)
        for key in ['semantics', 'labels', 'label', 'occ', 'occupancy']:
            if key in data:
                occ = data[key]
                break
        else:
            raise KeyError(f'{path} has no known occupancy key: {list(data.keys())}')
    else:
        raise ValueError(
            f'Unsupported occupancy file extension: {path}. '
            'FoundationSSC SemanticKITTI uses .npy dense occupancy files.')
    return occ.astype(np.int32, copy=False), path


def occupancy_to_bev(occ, empty_labels):
    """沿高度方向选最高有效体素类别，生成 BEV label map。"""
    if occ.ndim != 3:
        raise ValueError(f'Expected occupancy shape (H,W,D), got {occ.shape}')
    empty_labels = set(empty_labels)
    occupied = np.ones_like(occ, dtype=bool)
    for label in empty_labels:
        occupied &= (occ != label)
    has_occupied = occupied.any(axis=2)
    top_from_end = np.argmax(occupied[:, :, ::-1], axis=2)
    top_index = occ.shape[2] - 1 - top_from_end
    bev = np.take_along_axis(occ, top_index[:, :, None], axis=2).squeeze(2)
    bev[~has_occupied] = 0
    return bev


def render_frame(occ, infos, frame_idx, scene_name, occ_path, traj, args):
    """渲染单帧 BEV occupancy + 当前局部坐标系下的自车轨迹。"""
    bev = occupancy_to_bev(occ, args.empty_labels)
    xmin, ymin, _, xmax, ymax, _ = args.pc_range

    # 颜色索引：0 作为空白背景；1~19 对应 SemanticKITTI 19 个语义类。
    colors = np.vstack([np.array([[1.0, 1.0, 1.0, 1.0]]), SEMANTICKITTI_COLORS])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    ax.imshow(bev.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
              interpolation='nearest', cmap=cmap, norm=norm, alpha=0.88)

    ax.plot(traj[:frame_idx + 1, 0], traj[:frame_idx + 1, 1],
            '-o', color='#0066ff', linewidth=2, markersize=3, label='ego history')
    ax.plot(traj[frame_idx:, 0], traj[frame_idx:, 1],
            '-o', color='#ff6600', linewidth=2, markersize=3, label='ego future')
    ax.scatter([0], [0], marker='^', s=90, c='#00aa00', edgecolors='black',
               zorder=5, label='current ego')
    ax.arrow(0, 0, 3, 0, width=0.08, head_width=0.7, color='#00aa00', zorder=5)

    info = infos[frame_idx]
    ax.set_title(f'{scene_name} | frame={frame_idx}/{len(infos)-1} | '
                 f'token={info["token"]} | occ={osp.basename(occ_path)}')
    ax.set_xlabel('local x / forward (m)')
    ax.set_ylabel('local y / left (m)')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return rgb


def visualize_scene(scene_name, infos, args, data_root):
    """把一个 scene 的所有帧写成 BEV 视频。"""
    if args.max_frames > 0:
        infos = infos[:args.max_frames]
    if not infos:
        return
    scene_dir = osp.join(args.output_dir, scene_name)
    frames_dir = osp.join(scene_dir, 'frames')
    os.makedirs(scene_dir, exist_ok=True)
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    video_path = osp.join(scene_dir, 'bev.mp4')
    #* 位姿矩阵只和pkl元数据有关，先缓存一次；避免每渲染一帧都重复解析四元数和构造矩阵。
    poses = [pose_matrix(info) for info in infos]
    total = len(infos)

    writer = None
    try:
        iterator = enumerate(infos)
        for frame_idx, info in progress_iter(iterator, total=total, desc=f'Visualizing {scene_name}'):
            occ, occ_path = load_occupancy(info, scene_name, data_root)
            #* 每一帧都以“当前帧自车”为原点，因此轨迹需要变换到当前帧局部坐标系。
            traj = scene_trajectory_from_pose_cache(poses, frame_idx)
            rgb = render_frame(occ, infos, frame_idx, scene_name, occ_path, traj, args)
            if writer is None:
                height, width = rgb.shape[:2]
                writer = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create video: {video_path}')
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if args.save_frames:
                cv2.imwrite(osp.join(frames_dir, f'{frame_idx:06d}.png'),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    print(f'Video saved: {video_path}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_pkl(args.pkl)
    data_root = infer_data_root(data, args)
    infos_by_scene = data['infos']
    if args.scene:
        if args.scene not in infos_by_scene:
            raise KeyError(f'{args.scene} not found in pkl. Available: {list(infos_by_scene)[:10]}')
        scene_names = [args.scene]
    else:
        scene_names = list(infos_by_scene.keys())
    for scene_name in scene_names:
        visualize_scene(scene_name, infos_by_scene[scene_name], args, data_root)


if __name__ == '__main__':
    main()
