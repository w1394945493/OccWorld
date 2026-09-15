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
import sys
import time

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
    parser.add_argument('--check-occ-files', action='store_true',
                        help=('check whether each dense occupancy .npy exists; disabled by default '
                              'because per-frame file stat can be slow on network filesystems'))
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
    #* 使用 os.scandir 比 os.listdir + 多次路径处理更轻量；只读取目录项名称，不读取点云内容。
    tokens = sorted(
        osp.splitext(entry.name)[0] for entry in os.scandir(voxel_dir)
        if entry.is_file() and entry.name.endswith('.bin'))
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


def token_to_pose_index(token):
    """SemanticKITTI token/文件名转 poses.txt 行号。

    #! FoundationSSC 的 occupancy 文件名通常是 000000_1_1.npy、000005_1_1.npy ...
    #! 其中 000110 不是“第22个occupancy样本”的pose，而是 poses.txt 中第110行对应的原始帧。
    #! 因此位姿索引必须由 int(token) 得到，不能使用窗口内的 frame_idx/list index。
    """
    return int(token)


def build_history_trajs(poses_lidar, token_pose_indices, frame_idx, his_ts):
    """构造历史 his_ts 步相邻位移；按 occupancy token 对应的真实 pose 行号取位姿。

    frame_idx 是 tokens 列表中的位置，例如第22个可用occupancy帧；
    token_pose_indices[frame_idx] 才是 poses.txt 中的真实行号，例如 110。
    """
    token_indices = [max(0, frame_idx - i) for i in range(his_ts, -1, -1)]
    pose_indices = [token_pose_indices[i] for i in token_indices]
    current_pose_idx = token_pose_indices[frame_idx]
    positions = relative_positions_in_current(poses_lidar, current_pose_idx, pose_indices)
    return (positions[1:] - positions[:-1])[:, :2].astype(np.float32)


def build_future_trajs(poses_lidar, token_pose_indices, frame_idx, fut_ts):
    """构造未来 fut_ts 步相邻位移和有效mask；按可用occupancy帧序列向后取。"""
    max_token_idx = len(token_pose_indices) - 1
    token_indices = [min(max_token_idx, frame_idx + i) for i in range(fut_ts + 1)]
    pose_indices = [token_pose_indices[i] for i in token_indices]
    masks = np.asarray(
        [1.0 if frame_idx + i <= max_token_idx else 0.0 for i in range(1, fut_ts + 1)],
        dtype=np.float32)
    current_pose_idx = token_pose_indices[frame_idx]
    positions = relative_positions_in_current(poses_lidar, current_pose_idx, pose_indices)
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
    """加载 SemanticKITTI 数据并构造一个真实帧的最小 OccWorld 风格 info。

    #! 该函数适合单帧调试；批量构造连续帧时，main() 会先读取 tokens/poses，
    #! 再循环调用 build_frame_info_from_cache，避免每一帧都重复扫描目录和读取位姿。
    """
    ann_file = resolve_ann_file(data_root)
    sequence_dir = osp.join(data_root, 'sequences', sequence)
    tokens = list_frame_tokens(sequence_dir)
    poses_lidar = load_lidar_poses(data_root, sequence)
    token_pose_indices = [token_to_pose_index(token) for token in tokens]
    return build_frame_info_from_cache(
        data_root=data_root,
        ann_file=ann_file,
        sequence=sequence,
        tokens=tokens,
        token_pose_indices=token_pose_indices,
        poses_lidar=poses_lidar,
        frame_idx=frame_idx,
        fut_ts=fut_ts,
        his_ts=his_ts,
        cmd_thresh=cmd_thresh)


def build_frame_info_from_cache(
        data_root,
        ann_file,
        sequence,
        tokens,
        token_pose_indices,
        poses_lidar,
        frame_idx,
        fut_ts,
        his_ts,
        cmd_thresh,
        check_occ_files=False):
    """基于已缓存的 tokens/poses 构造单帧 info，避免批量构建时重复 IO。

    #* 性能关键点：
    #* - tokens 来自一次 os.scandir(<seq>/voxels)；
    #* - poses_lidar 来自一次 calib.txt/poses.txt 读取；
    #* - 连续 N 帧只在内存中按 frame_idx 取对应元素，不再重复扫描网络盘目录。
    """
    sequence_dir = osp.join(data_root, 'sequences', sequence)
    num_frames = min(len(tokens), len(token_pose_indices))
    if frame_idx < 0 or frame_idx >= num_frames:
        raise IndexError(f'frame_idx={frame_idx} out of range [0, {num_frames - 1}]')

    token = tokens[frame_idx]
    pose_idx = token_pose_indices[frame_idx]
    if pose_idx < 0 or pose_idx >= len(poses_lidar):
        raise IndexError(
            f'token={token} maps to pose index {pose_idx}, but poses.txt has '
            f'{len(poses_lidar)} poses.')
    prev_token = tokens[frame_idx - 1] if frame_idx > 0 else ''
    next_token = tokens[frame_idx + 1] if frame_idx + 1 < num_frames else ''
    scene_name = f'sequence-{sequence}'
    pose = poses_lidar[pose_idx]

    gt_ego_his_trajs = build_history_trajs(poses_lidar, token_pose_indices, frame_idx, his_ts)
    gt_ego_fut_trajs, gt_ego_fut_masks = build_future_trajs(
        poses_lidar, token_pose_indices, frame_idx, fut_ts)
    gt_ego_fut_cmd = build_pseudo_command(gt_ego_fut_trajs, cmd_thresh)

    #* 与 FoundationSSC 完全一致的 dense occupancy 路径：
    #*   ann_file / sequence / (frame_id + '_1_1.npy')
    voxel_path = osp.join(ann_file, sequence, f'{token}_1_1.npy')
    if check_occ_files and not osp.isfile(voxel_path):
        raise FileNotFoundError(
            f'Missing dense occupancy file: {voxel_path}. '
            'Expected FoundationSSC path <data-root>/labels/<seq>/<frame_id>_1_1.npy.')
    info = {
        #*==================== 1. 当前帧基础信息 ====================#
        'lidar_path': osp.join(sequence_dir, 'velodyne', f'{token}.bin'),  # 当前LiDAR点云路径
        #* 这里直接写入确定性路径，不默认逐帧 osp.isfile 检查；网络盘上大量 stat 会明显拖慢 converter。
        'voxel_path': voxel_path,  # FoundationSSC dense occupancy GT
        'occ_path': voxel_path,  # 可视化脚本优先读取该字段
        'token': token,  # 当前帧ID；例如000000
        'prev': prev_token,  # 同一sequence内上一帧token；首帧为空
        'next': next_token,  # 同一sequence内下一帧token；末帧为空
        'frame_idx': frame_idx,  # 当前帧在sequence内的整数编号
        'pose_idx': pose_idx,  # 当前token在poses.txt中的真实行号；例如token=000110 -> pose_idx=110
        'scene_token': scene_name,  # 用sequence名代替nuScenes scene token
        'timestamp': int(pose_idx),  # 最小示例用pose_idx作为伪时间戳，更贴近原始连续帧编号
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
        'fut_valid_flag': bool(frame_idx + fut_ts < num_frames),  # 是否有完整未来fut_ts个occupancy关键帧
        'cams': {},  # SemanticKITTI LiDAR-only最小样本可先置空
        'sweeps': [],  # 如不使用多帧sweep，可先置空
    }
    return info


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

    #*==================== 1. 确定数据根目录与输出scene名称 ====================#
    # SemanticKITTI 的数据根目录为 .../dataset，内部包含 sequences/ 和 labels/。
    # 本脚本保存成 OccWorld 风格 pkl 时，用 sequence-xx 作为 scene key。
    data_root = args.data_root
    sequence = args.sequence
    scene_name = f'sequence-{sequence}'
    sequence_dir = osp.join(data_root, 'sequences', sequence)

    #*==================== 2. 确定 FoundationSSC dense occupancy 根目录 ====================#
    # 这里不读取原始逐点 labels/*.label，而是读取 dense occupancy：
    #   <data_root>/labels/<sequence>/<frame_id>_1_1.npy
    ann_file = resolve_ann_file(data_root)

    #*==================== 3. 读取该sequence的可用occupancy帧ID ====================#
    # FoundationSSC 以 <data_root>/sequences/<sequence>/voxels/*.bin 来确定有哪些帧。
    # tokens 是有序字符串列表，例如 ['000000', '000005', ..., '000110']。
    tokens = list_frame_tokens(sequence_dir)

    #*==================== 4. 读取并转换自车位姿 ====================#
    # poses.txt 通常是 cam0->global；calib.txt 的 Tr 是 LiDAR->cam0。
    # load_lidar_poses 会得到每个原始帧的 LiDAR/ego->global 位姿。
    poses_lidar = load_lidar_poses(data_root, sequence)

    #! 非常关键：tokens列表下标不一定等于 poses.txt 行号。
    #! 例如第22个可用occupancy token 可能是 '000110'，其真实pose行号应为110，而不是22。
    #! 因此后续所有位姿都通过 token_pose_indices = int(token) 对齐。
    token_pose_indices = [token_to_pose_index(token) for token in tokens]
    if token_pose_indices and max(token_pose_indices) >= len(poses_lidar):
        raise IndexError(
            f'Max token pose index {max(token_pose_indices)} exceeds poses.txt length '
            f'{len(poses_lidar)}. Please check sequence {sequence} tokens/poses.')

    #*==================== 5. 确定要保存的连续窗口 ====================#
    # all_frames=True：保存该sequence的全部可用occupancy帧；
    # 否则：从 --frame-idx 指定的 tokens列表下标开始，保存 --num-frames 个连续可用帧。
    total_frames = len(tokens)
    if args.all_frames:
        start_frame_idx = 0
        end_frame_idx = total_frames
        print(f'Build all frames for {scene_name}.')
    else:
        if args.num_frames <= 0:
            raise ValueError(f'num_frames must be positive, got {args.num_frames}')
        start_frame_idx = args.frame_idx
        end_frame_idx = start_frame_idx + args.num_frames
        if start_frame_idx < 0 or start_frame_idx >= total_frames:
            raise IndexError(
                f'frame_idx={start_frame_idx} out of range [0, {total_frames - 1}]')
        if end_frame_idx > total_frames:
            raise IndexError(
                f'Requested window [{start_frame_idx}, {end_frame_idx}) exceeds '
                f'sequence length {total_frames}. Please reduce --num-frames or '
                f'use a smaller --frame-idx.')

    num_frames_to_build = end_frame_idx - start_frame_idx
    print(f'Loaded {scene_name}: {len(tokens)} occupancy frame tokens, '
          f'{len(poses_lidar)} poses.')
    print(f'Build window in token-list index: [{start_frame_idx}, {end_frame_idx}) '
          f'({num_frames_to_build} frames).')

    #*==================== 6. 逐帧构造 OccWorld 风格 frame_info ====================#
    # 每个 info 仍然只描述一帧：
    #   - occ_path / voxel_path 指向 dense occupancy GT；
    #   - ego2global_translation / rotation 保存当前帧自车位姿；
    #   - gt_ego_his_trajs / gt_ego_fut_trajs / pose_mode 构造自车运动监督；
    #   - cams / sweeps / bbox 等目前置空，作为 LiDAR-only occupancy forecasting 最小样本。
    infos = []
    t_start = time.time()
    for offset, frame_idx in enumerate(range(start_frame_idx, end_frame_idx), 1):
        info = build_frame_info_from_cache(
            data_root=data_root,
            ann_file=ann_file,
            sequence=sequence,
            tokens=tokens,
            token_pose_indices=token_pose_indices,
            poses_lidar=poses_lidar,
            frame_idx=frame_idx,
            fut_ts=args.fut_ts,
            his_ts=args.his_ts,
            cmd_thresh=args.cmd_thresh,
            check_occ_files=args.check_occ_files)
        infos.append(info)

        #* 长sequence构建时每100帧刷新一次进度，避免终端看起来无响应。
        if offset == 1 or offset == num_frames_to_build or offset % 100 == 0:
            elapsed = time.time() - t_start
            fps = offset / max(elapsed, 1e-6)
            print(f'\rBuilding infos: {offset}/{num_frames_to_build} '
                  f'({fps:.1f} frame/s)', end='', file=sys.stderr, flush=True)
    print('', file=sys.stderr)

    print(f'Built {len(infos)} infos for {scene_name}.')
    if infos:
        print_info(infos[0])

    #*==================== 7. 保存为 OccWorld 风格 pkl ====================#
    # 最终结构：
    #   {
    #       'infos': {'sequence-00': [frame_info_0, frame_info_1, ...]},
    #       'metadata': {...}
    #   }
    # 这样 visualize_semantickitti_pkl_bev.py 可以按 scene_name 取出连续帧并合成视频。
    if args.out_pkl:
        dump_minimal_pkl(
            infos=infos,
            scene_name=scene_name,
            out_pkl=args.out_pkl,
            data_root=data_root,
            sequence=sequence,
            start_frame_idx=start_frame_idx,
            num_frames=len(infos))
        print(f'Wrote minimal pkl to: {args.out_pkl}')


if __name__ == '__main__':
    main()
