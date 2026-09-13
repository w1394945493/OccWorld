#!/usr/bin/env python3
"""无界面抽取 OccWorld Dataset 时序样本，并生成 Occupancy/3D Box/自车轨迹 BEV 视频。"""

import argparse
import os
import os.path as osp
import sys

import cv2
os.environ.setdefault('MPLCONFIGDIR', '/tmp/occworld_matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import matplotlib

matplotlib.use('Agg')  # 无显示器/无X Server环境下使用离屏渲染后端
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from mmengine import Config
from pyquaternion import Quaternion


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Occ3D-nuScenes 17个占用语义类别的常用颜色；最后的白色对应free体素。
OCC_COLORS = np.asarray([
    [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
    [0, 255, 255], [255, 127, 0], [255, 0, 0], [255, 240, 150],
    [135, 60, 0], [160, 32, 240], [255, 0, 255], [139, 137, 137],
    [75, 0, 75], [150, 240, 80], [230, 230, 250], [0, 175, 0],
    [0, 191, 255], [255, 255, 255],
], dtype=np.float32) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='随机抽取OccWorld时序样本，离屏生成BEV检查图和MP4视频。')
    parser.add_argument('--py-config', required=True, help='包含train/val_dataset_config的配置文件')
    parser.add_argument('--split', choices=['train', 'val'], default='val', help='从训练集或验证集抽样')
    parser.add_argument('--dataset-index', type=int, default=-1,
                        help='Dataset索引；负数表示依据seed随机选择')
    parser.add_argument('--seed', type=int, default=42, help='控制Dataset内部随机窗口起点')
    parser.add_argument('--output', default='out/dataset_sequence_bev.mp4', help='输出MP4路径')
    parser.add_argument('--fps', type=float, default=2.0, help='输出视频帧率')
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                        help='Occupancy对应的LiDAR坐标范围')
    parser.add_argument('--free-label', type=int, default=17, help='空体素类别编号')
    parser.add_argument('--ignore-label', type=int, default=255, help='无效/忽略体素类别编号')
    parser.add_argument('--save-frames', action='store_true', help='同时保存逐帧PNG检查图')
    return parser.parse_args()


def transform_matrix(translation, rotation):
    """由[平移, wxyz四元数]构造局部坐标系到父坐标系的4x4变换。"""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def lidar_to_global(info):
    """根据元数据计算当前LIDAR_TOP坐标系到global坐标系的变换。"""
    ego_to_global = transform_matrix(info['ego2global_translation'], info['ego2global_rotation'])
    lidar_to_ego = transform_matrix(info['lidar2ego_translation'], info['lidar2ego_rotation'])
    return ego_to_global @ lidar_to_ego


def ego_to_global(info):
    """根据元数据构造当前ego坐标系到global坐标系的变换。"""
    return transform_matrix(info['ego2global_translation'], info['ego2global_rotation'])


def trajectory_in_current_lidar(infos, current_index):
    """将窗口内所有帧的LiDAR原点统一变换到当前帧LiDAR坐标系。"""
    lidar_to_globals = [lidar_to_global(info) for info in infos]
    global_to_current = np.linalg.inv(lidar_to_globals[current_index])
    origin = np.asarray([0.0, 0.0, 0.0, 1.0])
    return np.stack([(global_to_current @ pose @ origin)[:2] for pose in lidar_to_globals])


def trajectory_in_current_ego(infos, current_index):
    """将窗口内各帧ego原点统一表达在当前帧ego坐标系，以便与Occ3D栅格对齐。"""
    ego_to_globals = [ego_to_global(info) for info in infos]
    global_to_current_ego = np.linalg.inv(ego_to_globals[current_index])
    origin = np.asarray([0.0, 0.0, 0.0, 1.0])
    return np.stack([(global_to_current_ego @ pose @ origin)[:2] for pose in ego_to_globals])


def validate_trajectory_coordinates(infos):
    """用pkl自带的下一帧位移，校验位姿变换得到的轨迹是否处于同一坐标系。"""
    errors = []
    for frame_index, info in enumerate(infos[:-1]):
        gt_future = np.asarray(info.get('gt_ego_fut_trajs', []))
        if gt_future.ndim != 2 or gt_future.shape[0] == 0 or gt_future.shape[1] < 2:
            continue
        trajectory = trajectory_in_current_lidar(infos, frame_index)
        pose_step = trajectory[frame_index + 1] - trajectory[frame_index]
        errors.append(np.linalg.norm(pose_step - gt_future[0, :2]))
    return np.asarray(errors, dtype=np.float64)


def occupancy_to_bev(occupancy, free_label, ignore_label):
    """沿高度取最高的有效占用体素，生成便于核对位置关系的BEV语义图。"""
    if occupancy.ndim != 3:
        raise ValueError(f'期望单帧Occupancy形状为(H,W,D)，实际为{occupancy.shape}')
    occupied = (occupancy != free_label) & (occupancy != ignore_label)
    has_occupied = occupied.any(axis=2)
    top_from_end = np.argmax(occupied[:, :, ::-1], axis=2)
    top_index = occupancy.shape[2] - 1 - top_from_end
    bev = np.take_along_axis(occupancy, top_index[:, :, None], axis=2).squeeze(2)
    bev[~has_occupied] = free_label
    return bev


def valid_boxes(info):
    """读取当前帧LiDAR坐标系3D框，并尽量沿用数据集的有效框过滤规则。"""
    boxes = np.asarray(info.get('gt_boxes', np.empty((0, 7))))
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        return np.empty((0, 7), dtype=np.float32)
    valid = info.get('valid_flag')
    if valid is not None and np.asarray(valid).shape == (len(boxes),):
        boxes = boxes[np.asarray(valid, dtype=bool)]
    elif 'num_lidar_pts' in info and len(info['num_lidar_pts']) == len(boxes):
        boxes = boxes[np.asarray(info['num_lidar_pts']) > 0]
    return boxes


def box_bev_corners(box, lidar_to_ego_matrix):
    """生成LiDAR框的四角，并通过标定外参将其转换到当前ego坐标系。"""
    x, y, length, width, yaw = box[0], box[1], box[3], box[4], box[6]
    local = np.asarray([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2],
    ])
    rotation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    corners_lidar = local @ rotation.T + np.asarray([x, y])
    corners_lidar_h = np.concatenate([
        corners_lidar, np.full((4, 1), box[2]), np.ones((4, 1))], axis=1)
    return (lidar_to_ego_matrix @ corners_lidar_h.T).T[:, :2]


def render_frame(occupancy, info, infos, frame_index, args):
    """将Occupancy、3D框和自车轨迹统一到当前ego坐标系后绘制BEV。"""
    bev = occupancy_to_bev(occupancy, args.free_label, args.ignore_label)
    xmin, ymin, _, xmax, ymax, _ = args.pc_range
    cmap = ListedColormap(OCC_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(OCC_COLORS) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    # Occ3D栅格位于当前ego坐标系，数组前两维按(x,y)组织；转置后x显示为横轴、y显示为纵轴。
    # nuScenes ego坐标约定为+x向前、+y向左。bbox和轨迹原本位于LIDAR_TOP坐标，需通过外参转换。
    ax.imshow(bev.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
              interpolation='nearest', cmap=cmap, norm=norm, alpha=0.82)

    boxes = valid_boxes(info)
    current_lidar_to_ego = transform_matrix(info['lidar2ego_translation'], info['lidar2ego_rotation'])
    for box in boxes:
        corners = box_bev_corners(box, current_lidar_to_ego)
        corners = np.vstack([corners, corners[0]])
        ax.plot(corners[:, 0], corners[:, 1], color='black', linewidth=1.0)
        center_ego = current_lidar_to_ego @ np.asarray([box[0], box[1], box[2], 1.0])
        ax.plot(center_ego[0], center_ego[1], '.', color='black', markersize=2)

    trajectory = trajectory_in_current_ego(infos, frame_index)
    ax.plot(trajectory[:frame_index + 1, 0], trajectory[:frame_index + 1, 1],
            '-o', color='#0066ff', linewidth=2, markersize=4, label='ego history')
    ax.plot(trajectory[frame_index:, 0], trajectory[frame_index:, 1],
            '-o', color='#ff6600', linewidth=2, markersize=4, label='ego future')
    ax.scatter([0], [0], marker='^', s=90, c='#00aa00', edgecolors='black',
               zorder=5, label='current ego')
    ax.arrow(0, 0, 3, 0, width=0.08, head_width=0.7, color='#00aa00', zorder=5)

    token = info.get('token', 'unknown')
    ax.set_title(f"{args.split} | frame {frame_index + 1}/{len(infos)} | "
                 f"token={token[:12]} | boxes={len(boxes)}")
    ax.set_xlabel('ego x / forward (m)')
    ax.set_ylabel('ego y / left (m)')
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


def main():
    args = parse_args()
    # 延迟导入可让--help在未激活OccWorld/mmdet3d环境时正常使用；实际加载数据仍要求项目完整环境。
    from dataset import OPENOCC_DATASET

    np.random.seed(args.seed)  # 同时固定Dataset内部np.random.randint选择的窗口起点
    cfg = Config.fromfile(args.py_config)
    dataset_cfg = dict(cfg.train_dataset_config if args.split == 'train' else cfg.val_dataset_config)
    dataset_cfg['test_mode'] = False  # 本脚本直接按每帧info读取框，无需Dataset固定参考帧的评估元数据
    dataset = OPENOCC_DATASET.build(dataset_cfg, default_args={'nusc': None})

    if len(dataset) == 0:
        raise RuntimeError('Dataset长度为0，无法抽样。')
    dataset_index = args.dataset_index
    if dataset_index < 0:
        dataset_index = int(np.random.randint(0, len(dataset)))
    if dataset_index >= len(dataset):
        raise IndexError(f'dataset-index={dataset_index}超出[0,{len(dataset) - 1}]')

    input_occs, _, metas = dataset[dataset_index]  # 确实调用项目Dataset类完成随机数据包加载
    scene_name = metas['scene_name']
    window_start = int(metas['window_start'])
    infos = dataset.nusc_infos[scene_name][window_start:window_start + len(input_occs)]
    if len(infos) != len(input_occs):
        raise RuntimeError('元数据帧数与Occupancy帧数不一致。')

    output_path = osp.abspath(args.output)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    frames_dir = osp.splitext(output_path)[0] + '_frames'
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    print(f'Dataset type : {dataset.__class__.__name__}')
    print(f'Dataset index: {dataset_index}')
    print(f'Scene/window : {scene_name}, start={window_start}, frames={len(input_occs)}')
    print(f'Occupancy    : shape={input_occs.shape}, dtype={input_occs.dtype}, '
          f'labels={np.unique(input_occs).tolist()}')
    trajectory_errors = validate_trajectory_coordinates(infos)
    if len(trajectory_errors):
        print(f'Pose/GT check: mean_error={trajectory_errors.mean():.6f} m, '
              f'max_error={trajectory_errors.max():.6f} m')
        if trajectory_errors.max() > 1e-3:
            print('WARNING      : 位姿轨迹与gt_ego_fut_trajs不一致，请检查pkl的坐标约定或变换矩阵。')

    writer = None
    try:
        for frame_index, (occupancy, info) in enumerate(zip(input_occs, infos)):
            rgb = render_frame(occupancy, info, infos, frame_index, args)
            if writer is None:
                height, width = rgb.shape[:2]
                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f'无法创建视频{output_path}，请检查OpenCV/FFmpeg编码支持。')
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if args.save_frames:
                cv2.imwrite(osp.join(frames_dir, f'{frame_index:03d}.png'),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()

    print(f'Video saved  : {output_path}')
    if args.save_frames:
        print(f'Frames saved : {frames_dir}')


if __name__ == '__main__':
    main()
