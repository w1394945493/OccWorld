#!/usr/bin/env python3
"""SemanticKITTI -> OccWorld 风格最小 pkl 元数据构造示例。

当前脚本完成“一个 SemanticKITTI 场景/片段”的最小 info 构建：
- 读取 FoundationSSC 使用的 SemanticKITTI 组织：
  <data_root>/sequences/<seq>/voxels/*.bin         # 用于确定帧 id
  <ann_file>/<seq>/<frame_id>_1_1.npy              # dense occupancy GT
  <data_root>/sequences/<seq>/poses.txt
  <data_root>/sequences/<seq>/calib.txt
- 根据 poses.txt/calib.txt 计算每帧 LiDAR 在 global 中的位姿；
- 为指定 sequence/frame 或指定连续窗口构造 OccWorld 风格 frame_info；
- 可选保存一个最小 pkl：
  {"infos": {"sequence-00": [frame_info, ...]}, "metadata": {...}}

注意：
1) 这个最小 info 主要面向 occupancy forecasting，不包含 3D bbox / agent 轨迹；
2) 这里不使用 SemanticKITTI 原始逐点 labels/*.label，而是使用 FoundationSSC 配置中的
   ann_file=<data_root>/labels，确定性读取 <ann_file>/<seq>/<frame_id>_1_1.npy。
"""

import argparse
import os
import os.path as osp
import pickle

import numpy as np


DEFAULT_DATA_ROOT = '/c20250502/wangyushen/Datasets/kitti/semantickitti/dataset'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build one minimal OccWorld-style SemanticKITTI frame_info.')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                        help='SemanticKITTI dataset root, usually ending with /dataset')
    parser.add_argument('--sequence', default='00', help='SemanticKITTI sequence id, e.g. 00')
    parser.add_argument('--frame-idx', type=int, default=0,
                        help='start frame index in the sequence')
    parser.add_argument('--num-frames', type=int, default=1,
                        help=('number of consecutive frames to save from --frame-idx; '
                              'ignored when --all-frames is set'))
    parser.add_argument('--all-frames', action='store_true',
                        help='build infos for all frames in the sequence instead of a fixed window')
    parser.add_argument('--fut-ts', type=int, default=6, help='future steps, default 6')
    parser.add_argument('--his-ts', type=int, default=2, help='history steps, default 2')
    parser.add_argument('--cmd-thresh', type=float, default=2.0,
                        help='lateral threshold in meters for pseudo command')
    parser.add_argument('--out-pkl', default='',
                        help='optional output pkl path for a minimal infos dict')
    return parser.parse_args()


def resolve_ann_file(data_root):
    """由数据集根目录确定 FoundationSSC dense occupancy 根目录。"""
    ann_file = osp.join(data_root, 'labels')
    if not osp.isdir(ann_file):
        raise FileNotFoundError(
            f'Cannot infer ann_file from data_root. Expected directory: {ann_file}')
    return ann_file


def read_calib(calib_path):
    """读取 SemanticKITTI calib.txt，并返回形如 {'Tr': 4x4, 'P0': ...} 的字典。"""
    if not osp.isfile(calib_path):
        raise FileNotFoundError(f'Missing calib file: {calib_path}')
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.split(':', 1)
            values = np.fromstring(value, sep=' ', dtype=np.float64)
            if values.size == 12:
                mat = np.eye(4, dtype=np.float64)
                mat[:3, :4] = values.reshape(3, 4)
            elif values.size == 16:
                mat = values.reshape(4, 4)
            else:
                mat = values
            calib[key] = mat
    if 'Tr' not in calib:
        raise KeyError(f'{calib_path} does not contain Tr calibration.')
    return calib


def read_poses(poses_path):
    """读取 poses.txt，返回每帧 4x4 位姿矩阵列表。"""
    if not osp.isfile(poses_path):
        raise FileNotFoundError(f'Missing poses file: {poses_path}')
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            values = np.fromstring(line, sep=' ', dtype=np.float64)
            if values.size != 12:
                raise ValueError(f'Invalid pose line in {poses_path}: {line}')
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = values.reshape(3, 4)
            poses.append(mat)
    return poses


def rotation_matrix_to_quaternion_wxyz(rot):
    """将3x3旋转矩阵转换为 wxyz 四元数，避免额外依赖 scipy/pyquaternion。"""
    rot = np.asarray(rot, dtype=np.float64)
    trace = np.trace(rot)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(rot)))
        if axis == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return quat.tolist()


def list_frame_tokens(sequence_dir):
    """按 FoundationSSC 逻辑根据 voxels/*.bin 获取有序帧 token。

    FoundationSSC 的 SemanticKITTIDataset.load_annotations() 默认扫描：
      <data_root>/sequences/<seq>/voxels/*.bin
    然后将同名 frame_id 映射到 dense occupancy：
      <ann_file>/<seq>/<frame_id>_1_1.npy
    """
    voxel_dir = osp.join(sequence_dir, 'voxels')
    if not osp.isdir(voxel_dir):
        raise FileNotFoundError(f'Missing voxels directory: {voxel_dir}')
    tokens = sorted(
        osp.splitext(name)[0] for name in os.listdir(voxel_dir)
        if name.endswith('.bin'))
    if not tokens:
        raise RuntimeError(f'No .bin files found in {voxel_dir}')
    return tokens


def load_lidar_poses(data_root, sequence):
    """读取 calib/poses，并将 SemanticKITTI pose 转为 LiDAR/ego -> global 位姿。

    SemanticKITTI 的 poses.txt 通常表示 cam0 到世界的位姿；calib.txt 中 Tr 表示
    velodyne -> cam0。因此：
        T_global_lidar = T_global_cam0 @ T_cam0_lidar
    这里把 LiDAR 坐标系直接当作 ego 坐标系，后续 lidar2ego 使用单位外参。
    """
    sequence_dir = osp.join(data_root, 'sequences', sequence)
    calib = read_calib(osp.join(sequence_dir, 'calib.txt'))
    poses_cam = read_poses(osp.join(sequence_dir, 'poses.txt'))
    tr_lidar_to_cam = calib['Tr']
    poses_lidar = [pose_cam @ tr_lidar_to_cam for pose_cam in poses_cam]
    return poses_lidar


def relative_positions_in_current(poses_lidar, current_idx, indices):
    """把若干帧 LiDAR 原点统一变换到 current_idx 帧 LiDAR 坐标系。"""
    global_to_current = np.linalg.inv(poses_lidar[current_idx])
    origin = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    points = []
    for idx in indices:
        point = global_to_current @ poses_lidar[idx] @ origin
        points.append(point[:3])
    return np.asarray(points, dtype=np.float64)


def build_history_trajs(poses_lidar, frame_idx, his_ts):
    """构造历史 his_ts 步相邻位移；场景开头不足时用最早帧重复补齐。"""
    indices = [max(0, frame_idx - i) for i in range(his_ts, -1, -1)]
    positions = relative_positions_in_current(poses_lidar, frame_idx, indices)
    return (positions[1:] - positions[:-1])[:, :2].astype(np.float32)


def build_future_trajs(poses_lidar, frame_idx, fut_ts):
    """构造未来 fut_ts 步相邻位移和有效mask；场景末尾不足时用最后帧补齐。"""
    max_idx = len(poses_lidar) - 1
    indices = [min(max_idx, frame_idx + i) for i in range(fut_ts + 1)]
    masks = np.asarray(
        [1.0 if frame_idx + i <= max_idx else 0.0 for i in range(1, fut_ts + 1)],
        dtype=np.float32)
    positions = relative_positions_in_current(poses_lidar, frame_idx, indices)
    fut = (positions[1:] - positions[:-1])[:, :2].astype(np.float32)
    fut[masks == 0] = 0.0
    return fut, masks


def build_pseudo_command(fut_trajs, cmd_thresh):
    """根据未来累计终点横向偏移构造 [右转, 左转, 直行] one-hot 伪指令。"""
    final_xy = np.cumsum(fut_trajs[:, :2], axis=0)[-1]
    lateral = final_xy[1]
    if lateral <= -cmd_thresh:
        return np.asarray([1, 0, 0], dtype=np.float32)  # right
    if lateral >= cmd_thresh:
        return np.asarray([0, 1, 0], dtype=np.float32)  # left
    return np.asarray([0, 0, 1], dtype=np.float32)  # straight


def build_minimal_frame_info(
        data_root,
        sequence='00',
        frame_idx=0,
        fut_ts=6,
        his_ts=2,
        cmd_thresh=2.0):
    """加载 SemanticKITTI 数据并构造一个真实帧的最小 OccWorld 风格 info。"""
    ann_file = resolve_ann_file(data_root)
    sequence_dir = osp.join(data_root, 'sequences', sequence)
    tokens = list_frame_tokens(sequence_dir)
    poses_lidar = load_lidar_poses(data_root, sequence)
    num_frames = min(len(tokens), len(poses_lidar))
    if frame_idx < 0 or frame_idx >= num_frames:
        raise IndexError(f'frame_idx={frame_idx} out of range [0, {num_frames - 1}]')

    token = tokens[frame_idx]
    prev_token = tokens[frame_idx - 1] if frame_idx > 0 else ''
    next_token = tokens[frame_idx + 1] if frame_idx + 1 < num_frames else ''
    scene_name = f'sequence-{sequence}'
    pose = poses_lidar[frame_idx]

    gt_ego_his_trajs = build_history_trajs(poses_lidar, frame_idx, his_ts)
    gt_ego_fut_trajs, gt_ego_fut_masks = build_future_trajs(poses_lidar, frame_idx, fut_ts)
    gt_ego_fut_cmd = build_pseudo_command(gt_ego_fut_trajs, cmd_thresh)

    #* 与 FoundationSSC 完全一致的 dense occupancy 路径：
    #*   ann_file / sequence / (frame_id + '_1_1.npy')
    voxel_path = osp.join(ann_file, sequence, f'{token}_1_1.npy')
    info = {
        #*==================== 1. 当前帧基础信息 ====================#
        'lidar_path': osp.join(sequence_dir, 'velodyne', f'{token}.bin'),  # 当前LiDAR点云路径
        'voxel_path': voxel_path if osp.isfile(voxel_path) else None,  # FoundationSSC dense occupancy GT
        'occ_path': voxel_path if osp.isfile(voxel_path) else None,  # 可视化脚本优先读取该字段
        'token': token,  # 当前帧ID；例如000000
        'prev': prev_token,  # 同一sequence内上一帧token；首帧为空
        'next': next_token,  # 同一sequence内下一帧token；末帧为空
        'frame_idx': frame_idx,  # 当前帧在sequence内的整数编号
        'scene_token': scene_name,  # 用sequence名代替nuScenes scene token
        'timestamp': int(frame_idx),  # 最小示例用frame_idx作为伪时间戳
        'map_location': 'semantickitti',  # 占位字段，保持接口兼容

        #*==================== 2. 当前帧位姿和标定 ====================#
        'lidar2ego_translation': [0.0, 0.0, 0.0],  # 简化：LiDAR系等同ego系
        'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],  # wxyz单位四元数
        'ego2global_translation': pose[:3, 3].astype(np.float64).tolist(),  # 当前LiDAR/ego在global中的位置
        'ego2global_rotation': rotation_matrix_to_quaternion_wxyz(pose[:3, :3]),  # 当前LiDAR/ego姿态

        #*==================== 3. 自车历史/未来轨迹标签 ====================#
        'gt_ego_his_trajs': gt_ego_his_trajs,  # (his_ts,2)，历史逐步位移
        'gt_ego_fut_trajs': gt_ego_fut_trajs,  # (fut_ts,2)，未来逐步位移
        'gt_ego_fut_masks': gt_ego_fut_masks,  # (fut_ts,)，未来有效性
        'gt_ego_fut_cmd': gt_ego_fut_cmd,  # (3,)，[右转, 左转, 直行]
        'pose_mode': gt_ego_fut_cmd.copy(),  # OccWorld dataset实际读取该字段

        #*==================== 4. 可选/占位字段 ====================#
        'fut_valid_flag': bool(frame_idx + fut_ts < num_frames),  # 是否有完整未来fut_ts帧
        'cams': {},  # SemanticKITTI LiDAR-only最小样本可先置空
        'sweeps': [],  # 如不使用多帧sweep，可先置空
    }
    return info


def build_sequence_infos(data_root, sequence, fut_ts, his_ts, cmd_thresh):
    """为一个 sequence 的所有 FoundationSSC 帧构造最小 info 列表。"""
    tokens = list_frame_tokens(osp.join(data_root, 'sequences', sequence))
    return build_window_infos(
        data_root=data_root,
        sequence=sequence,
        start_frame_idx=0,
        num_frames=len(tokens),
        fut_ts=fut_ts,
        his_ts=his_ts,
        cmd_thresh=cmd_thresh)


def build_window_infos(
        data_root,
        sequence,
        start_frame_idx,
        num_frames,
        fut_ts,
        his_ts,
        cmd_thresh):
    """为一个 sequence 中的连续窗口构造 info 列表。

    #! 这里的“场景”不是重新切分 SemanticKITTI，而是在同一个 sequence 内截取
    #! [start_frame_idx, start_frame_idx + num_frames) 这段连续帧保存到同一个
    #! pkl scene key 下，方便 OccWorld 风格 dataset/可视化按时间顺序读取。
    #! 每个 info 仍然是“单帧元数据”，列表顺序就是后续视频可视化和时序建模顺序。
    """
    if num_frames <= 0:
        raise ValueError(f'num_frames must be positive, got {num_frames}')
    tokens = list_frame_tokens(osp.join(data_root, 'sequences', sequence))
    poses_lidar = load_lidar_poses(data_root, sequence)
    total_frames = min(len(tokens), len(poses_lidar))
    end_frame_idx = start_frame_idx + num_frames
    if start_frame_idx < 0 or start_frame_idx >= total_frames:
        raise IndexError(
            f'frame_idx={start_frame_idx} out of range [0, {total_frames - 1}]')
    if end_frame_idx > total_frames:
        raise IndexError(
            f'Requested window [{start_frame_idx}, {end_frame_idx}) exceeds '
            f'sequence length {total_frames}. Please reduce --num-frames or '
            f'use a smaller --frame-idx.')
    return [
        build_minimal_frame_info(
            data_root=data_root,
            sequence=sequence,
            frame_idx=i,
            fut_ts=fut_ts,
            his_ts=his_ts,
            cmd_thresh=cmd_thresh)
        for i in range(start_frame_idx, end_frame_idx)
    ]


def dump_minimal_pkl(
        infos,
        scene_name,
        out_pkl,
        data_root,
        sequence,
        start_frame_idx,
        num_frames):
    """保存最小 pkl，infos 可以是一帧、一个连续窗口或完整 sequence。"""
    os.makedirs(osp.dirname(osp.abspath(out_pkl)), exist_ok=True)
    data = {
        'infos': {scene_name: infos},
        'metadata': {
            'dataset': 'SemanticKITTI',
            'version': 'foundation_ssc_style_minimal',
            'data_root': osp.abspath(data_root),
            'ann_file': osp.join(osp.abspath(data_root), 'labels'),
            'sequence': sequence,
            'start_frame_idx': int(start_frame_idx),
            'num_frames': int(num_frames),
            'note': 'Dense occupancy path follows FoundationSSC: ann_file/seq/frame_id_1_1.npy.',
        },
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)


def print_info(info):
    """紧凑打印 info，数组只打印shape/dtype和值。"""
    print('Minimal SemanticKITTI frame_info:')
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            print(f'{key}: shape={value.shape}, dtype={value.dtype}, value={value.tolist()}')
        else:
            print(f'{key}: {value}')


def main():
    args = parse_args()
    scene_name = f'sequence-{args.sequence}'
    if args.all_frames:
        infos = build_sequence_infos(
            data_root=args.data_root,
            sequence=args.sequence,
            fut_ts=args.fut_ts,
            his_ts=args.his_ts,
            cmd_thresh=args.cmd_thresh)
        print(f'Built {len(infos)} infos for {scene_name}')
        print_info(infos[0])
        start_frame_idx = 0
    else:
        infos = build_window_infos(
            data_root=args.data_root,
            sequence=args.sequence,
            start_frame_idx=args.frame_idx,
            num_frames=args.num_frames,
            fut_ts=args.fut_ts,
            his_ts=args.his_ts,
            cmd_thresh=args.cmd_thresh)
        print(f'Built {len(infos)} consecutive infos for {scene_name}: '
              f'[{args.frame_idx}, {args.frame_idx + args.num_frames})')
        print_info(infos[0])
        start_frame_idx = args.frame_idx
    if args.out_pkl:
        dump_minimal_pkl(
            infos=infos,
            scene_name=scene_name,
            out_pkl=args.out_pkl,
            data_root=args.data_root,
            sequence=args.sequence,
            start_frame_idx=start_frame_idx,
            num_frames=len(infos))
        print(f'Wrote minimal pkl to: {args.out_pkl}')


if __name__ == '__main__':
    main()
