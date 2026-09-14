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


# 与visualize_demo.py及config/label_mapping/nuscenes-occ.yaml保持一致：
# 0=others；1~16依次为barrier~vegetation；17=empty/free。
# 官方demo绘制时直接过滤掉0和17，因此其颜色表只显式列出了中间的1~16类；
# 这里使用Matplotlib按原始类别编号索引颜色，必须在首尾分别补上others和empty颜色。
OCC_COLORS = np.asarray([
    [0, 0, 0],        # 0: others（官方demo不显示）
    [255, 120, 50],   # 1: barrier
    [255, 192, 203],  # 2: bicycle
    [255, 255, 0],    # 3: bus
    [0, 150, 245],    # 4: car
    [0, 255, 255],    # 5: construction_vehicle
    [255, 127, 0],    # 6: motorcycle
    [255, 0, 0],      # 7: pedestrian
    [255, 240, 150],  # 8: traffic_cone
    [135, 60, 0],     # 9: trailer
    [160, 32, 240],   # 10: truck
    [255, 0, 255],    # 11: driveable_surface
    [139, 137, 137],  # 12: other_flat
    [75, 0, 75],      # 13: sidewalk
    [150, 240, 80],   # 14: terrain
    [230, 230, 250],  # 15: manmade
    [0, 175, 0],      # 16: vegetation
    [255, 255, 255],  # 17: empty/free（BEV背景）
], dtype=np.float32) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='随机抽取OccWorld时序样本，离屏生成BEV检查图和MP4视频。')
    parser.add_argument('--py-config', required=True, help='包含train/val_dataset_config的配置文件')
    parser.add_argument('--split', choices=['train', 'val'], default='val', help='从训练集或验证集抽样')
    parser.add_argument('--dataset-index', type=int, default=-1,
                        help='Dataset索引；负数表示依据seed随机选择')
    parser.add_argument('--seed', type=int, default=42, help='控制Dataset内部随机窗口起点')
    parser.add_argument('--motion', choices=['any', 'turn'], default='any',
                        help='any随机抽样；turn优先选择自车轨迹发生明显转向的窗口')
    parser.add_argument('--min-turn-deg', type=float, default=10.0,
                        help='turn模式的最小首尾航向变化，单位度')
    parser.add_argument('--min-lateral-m', type=float, default=3.0,
                        help='turn模式的最小累计横向偏移，单位m；满足两个阈值之一即可')
    parser.add_argument('--output', default='out/dataset_sequence_bev.mp4', help='输出MP4路径')
    parser.add_argument('--fps', type=float, default=2.0, help='输出视频帧率')
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                        help='Occupancy对应的LiDAR坐标范围')
    parser.add_argument('--free-label', type=int, default=17, help='空体素类别编号')
    parser.add_argument('--unknown-label', type=int, default=0,
                        help='others/unknown类别编号；与官方demo一致，不作为有效占用显示')
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


def turn_metrics(infos):
    """计算窗口末帧相对首帧ego坐标系的横向偏移和航向变化。"""
    first_to_global = ego_to_global(infos[0])
    last_to_global = ego_to_global(infos[-1])
    last_to_first = np.linalg.inv(first_to_global) @ last_to_global
    lateral_m = abs(float(last_to_first[1, 3]))
    turn_deg = abs(float(np.degrees(np.arctan2(last_to_first[1, 0], last_to_first[0, 0]))))
    return lateral_m, turn_deg


def select_turn_window(dataset, rng, dataset_index, min_lateral_m, min_turn_deg):
    """只扫描轻量级pkl位姿，从所有合法连续窗口中随机选择明显转向片段。"""
    scene_indices = range(len(dataset.scene_names))
    if dataset_index >= 0:
        scene_indices = [dataset_index % len(dataset.scene_names)]
    candidates = []
    window_len = dataset.return_len
    required_len = dataset.return_len + dataset.offset
    for scene_index in scene_indices:
        scene_name = dataset.scene_names[scene_index]
        scene_infos = dataset.nusc_infos[scene_name]
        for window_start in range(len(scene_infos) - required_len + 1):
            infos = scene_infos[window_start:window_start + window_len]
            lateral_m, turn_deg = turn_metrics(infos)
            if lateral_m >= min_lateral_m or turn_deg >= min_turn_deg:
                candidates.append((scene_index, scene_name, window_start, lateral_m, turn_deg))
    if not candidates:
        raise RuntimeError(
            f'没有找到转向窗口：lateral>={min_lateral_m}m或turn>={min_turn_deg}deg；'
            '请适当降低--min-lateral-m/--min-turn-deg。')
    return candidates[int(rng.randint(0, len(candidates)))], len(candidates)


def load_input_occupancies(dataset, scene_name, window_start):
    """按照Dataset路径约定加载一个已经确定起点的连续Occupancy窗口。"""
    occs = []
    for info in dataset.nusc_infos[scene_name][window_start:window_start + dataset.return_len]:
        label_file = osp.join(
            dataset.occ_path, dataset.input_dataset, scene_name, info['token'], 'labels.npz')
        with np.load(label_file) as label:
            occs.append(label['semantics'])
    return np.stack(occs).astype(np.int64, copy=False)


def occupancy_to_bev(occupancy, free_label, unknown_label, ignore_label):
    """沿高度取最高的有效占用体素，生成便于核对位置关系的BEV语义图。"""
    if occupancy.ndim != 3:
        raise ValueError(f'期望单帧Occupancy形状为(H,W,D)，实际为{occupancy.shape}')
    # 官方visualize_demo.py仅显示0<label<17，即排除others(0)、empty(17)和ignore(255)。
    occupied = ((occupancy != free_label) & (occupancy != unknown_label)
                & (occupancy != ignore_label))
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
    # *==================== 3D Occupancy投影为2D BEV ====================#
    # occupancy形状为(H,W,D)，每个体素保存一个语义类别编号；D表示离散高度层。
    # occupancy_to_bev()沿高度方向检查每个(x,y)位置，并选择最高处的有效占用体素类别，
    # 同时排除empty/free、others/unknown和ignore体素；整列均无有效占用时填入free_label。
    # 返回bev形状为(H,W)，供后续使用不同颜色绘制俯视语义图。这里只压缩高度维，
    # 不执行LiDAR/ego坐标变换；该Occupancy本身已经位于当前帧ego坐标系。
    bev = occupancy_to_bev(occupancy, args.free_label, args.unknown_label, args.ignore_label)
    xmin, ymin, _, xmax, ymax, _ = args.pc_range
    cmap = ListedColormap(OCC_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(OCC_COLORS) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    # Occ3D栅格位于当前ego坐标系，数组前两维按(x,y)组织；转置后x显示为横轴、y显示为纵轴。
    # nuScenes ego坐标约定为+x向前、+y向左。bbox和轨迹原本位于LIDAR_TOP坐标，需通过外参转换。
    ax.imshow(bev.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
              interpolation='nearest', cmap=cmap, norm=norm, alpha=0.82)

    # *==================== 3D BBox转换到Occupancy所在的ego坐标系 ====================#
    # 从当前帧info中读取LiDAR坐标系下的GT框，并使用valid_flag或num_lidar_pts过滤无效框。
    # boxes通常形如(N,7)，每个框为[x,y,z,length,width,height,yaw]。
    boxes = valid_boxes(info)
    # 根据当前帧LIDAR_TOP的标定平移和旋转构造齐次矩阵T_lidar_to_ego；
    # 后续用它把LiDAR坐标系中的框统一转换到Occ3D使用的当前ego坐标系。
    current_lidar_to_ego = transform_matrix(info['lidar2ego_translation'], info['lidar2ego_rotation'])
    for box in boxes:  # 逐个绘制当前帧的有效3D框
        # 先根据框中心、长宽和yaw计算LiDAR BEV四角，再用T_lidar_to_ego转换为ego坐标，输出(4,2)。
        corners = box_bev_corners(box, current_lidar_to_ego)
        # 在末尾重复第一个角点，将4个角点闭合为“角1→角2→角3→角4→角1”的矩形折线。
        corners = np.vstack([corners, corners[0]])
        ax.plot(corners[:, 0], corners[:, 1], color='black', linewidth=1.0)  # 绘制ego坐标系下的框轮廓
        # 将LiDAR框中心写成齐次坐标[x,y,z,1]并左乘外参，得到ego坐标系下的三维中心。
        center_ego = current_lidar_to_ego @ np.asarray([box[0], box[1], box[2], 1.0])
        ax.plot(center_ego[0], center_ego[1], '.', color='black', markersize=2)  # 在BEV中标出框中心

    # *==================== 自车历史/未来轨迹坐标统一 ====================#
    # infos包含当前连续窗口内的全部帧；每帧均提供该时刻ego坐标系到global坐标系的位姿。
    # trajectory_in_current_ego()先把各帧ego原点变换到global，再统一变换到第frame_index帧的
    # 当前ego坐标系，使轨迹与当前帧Occ3D Occupancy使用相同坐标约定：+x向前、+y向左。
    # 返回trajectory形状为(F,2)，其中F为窗口帧数，每行是对应时刻相对当前自车的(x,y)位置；
    # trajectory[frame_index]理论上为(0,0)，之前的点用于绘制历史，当前点及之后的点用于绘制未来。
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
    rng = np.random.RandomState(args.seed)  # 单独用于转向候选窗口的可复现随机选择
    cfg = Config.fromfile(args.py_config)
    dataset_cfg = dict(cfg.train_dataset_config if args.split == 'train' else cfg.val_dataset_config)
    dataset_cfg['test_mode'] = False  # 本脚本直接按每帧info读取框，无需Dataset固定参考帧的评估元数据
    dataset = OPENOCC_DATASET.build(dataset_cfg, default_args={'nusc': None})

    if len(dataset) == 0:
        raise RuntimeError('Dataset长度为0，无法抽样。')
    dataset_index = args.dataset_index
    selection_text = 'random'
    if args.motion == 'turn':
        selected, candidate_count = select_turn_window(
            dataset, rng, dataset_index, args.min_lateral_m, args.min_turn_deg)
        scene_index, scene_name, window_start, lateral_m, turn_deg = selected
        dataset_index = scene_index
        input_occs = load_input_occupancies(dataset, scene_name, window_start)
        selection_text = (f'turn ({candidate_count} candidates, lateral={lateral_m:.2f}m, '
                          f'heading_change={turn_deg:.1f}deg)')
    else:
        if dataset_index < 0:
            dataset_index = int(np.random.randint(0, len(dataset)))
        if dataset_index >= len(dataset):
            raise IndexError(f'dataset-index={dataset_index}超出[0,{len(dataset) - 1}]')
        input_occs, _, metas = dataset[dataset_index]  # any模式沿用Dataset原始随机窗口逻辑
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
    print(f'Selection    : {selection_text}')
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
