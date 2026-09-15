#!/usr/bin/env python3
"""离线推理OccWorld，并以BEV视角可视化未来Occupancy、GT轨迹和预测轨迹。

该脚本结合了两个现有脚本的用途：
1) 参考 visualize_demo.py：按验证集scene_idx选择样本，加载模型并执行forward_autoreg_with_pose；
2) 参考 tools/visualize_dataset_sequence_bev.py：将3D Occupancy压缩为BEV语义图，并离屏保存视频。

输出视频每帧包含两个子图：
- 左：未来GT Occupancy；
- 右：模型预测未来Occupancy；
两侧均叠加GT自车未来轨迹和预测自车未来轨迹。
"""

import argparse
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/occworld_matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import matplotlib

matplotlib.use('Agg')  # 无界面服务器使用离屏渲染后端
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
from mmengine.runner import set_random_seed


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# 与visualize_demo.py颜色表保持一致。0=others，1~16为nuScenes语义类，17=empty/free。
OCC_COLORS = np.asarray([
    [0, 0, 0],        # 0: others/unknown
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
    [255, 255, 255],  # 17: empty/free
], dtype=np.float32) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='OccWorld自回归预测结果BEV可视化：GT/Pred Occupancy + GT/Pred自车轨迹。')
    parser.add_argument('--py-config', required=True, help='OccWorld配置文件，如config/occworld_custom.py')
    parser.add_argument('--work-dir', required=True, help='实验目录；默认从其中读取latest.pth')
    parser.add_argument('--resume-from', type=str, default='', help='指定checkpoint；为空时优先使用work_dir/latest.pth')
    parser.add_argument('--dir-name', type=str, default='vis_autoreg_bev', help='输出子目录名')
    parser.add_argument('--scene-idx', nargs='+', type=int,
                        default=[6, 7, 16, 18, 19, 87, 89, 96, 101],
                        help='验证集dataloader中的样本索引；不是nuScenes scene-xxxx编号')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--fps', type=float, default=2.0, help='输出视频帧率')
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                        help='Occupancy栅格对应的ego坐标范围')
    parser.add_argument('--free-label', type=int, default=17, help='empty/free类别编号')
    parser.add_argument('--unknown-label', type=int, default=0, help='others/unknown类别编号')
    parser.add_argument('--ignore-label', type=int, default=255, help='ignore类别编号')
    parser.add_argument('--save-frames', action='store_true', help='同时保存逐帧PNG')
    return parser.parse_args()


def to_numpy(x):
    """兼容torch.Tensor / numpy.ndarray / list，统一转为CPU numpy。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def occupancy_to_bev(occupancy, free_label, unknown_label, ignore_label):
    """将单帧(H,W,D)语义Occupancy沿高度维压缩为BEV语义图。"""
    if occupancy.ndim != 3:
        raise ValueError(f'期望单帧Occupancy形状为(H,W,D)，实际为{occupancy.shape}')
    #! 与visualize_demo.py保持一致：仅显示有效占用语义类，过滤others/empty/ignore。
    occupied = ((occupancy != free_label) & (occupancy != unknown_label)
                & (occupancy != ignore_label))
    has_occupied = occupied.any(axis=2)
    top_from_end = np.argmax(occupied[:, :, ::-1], axis=2)
    top_index = occupancy.shape[2] - 1 - top_from_end
    bev = np.take_along_axis(occupancy, top_index[:, :, None], axis=2).squeeze(2)
    bev[~has_occupied] = free_label
    return bev


def cumulative_trajectory(step_xy):
    """将未来逐步位移(dx,dy)转换为从当前自车原点出发的累计轨迹。"""
    step_xy = np.asarray(step_xy, dtype=np.float64)
    if step_xy.ndim == 3:
        step_xy = step_xy[0]
    if step_xy.size == 0:
        return np.zeros((1, 2), dtype=np.float64)
    cum = np.cumsum(step_xy[:, :2], axis=0)
    return np.concatenate([np.zeros((1, 2), dtype=np.float64), cum], axis=0)


def draw_bev(ax, occupancy, title, gt_traj, pred_traj, frame_id, args):
    """在一个子图中绘制BEV Occupancy，并叠加GT/Pred未来轨迹。"""
    bev = occupancy_to_bev(occupancy, args.free_label, args.unknown_label, args.ignore_label)
    xmin, ymin, _, xmax, ymax, _ = args.pc_range
    cmap = ListedColormap(OCC_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(OCC_COLORS) + 0.5), cmap.N)

    #! Occupancy数组前两维按(x,y)组织；转置后横轴为ego x/forward，纵轴为ego y/left。
    ax.imshow(bev.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
              interpolation='nearest', cmap=cmap, norm=norm, alpha=0.86)

    #* 轨迹坐标：gt_traj/pred_traj均为从当前自车(0,0)出发的累计未来位置。
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], '--', color='#00aa00',
            linewidth=2.2, label='GT ego traj')
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-', color='#ff6600',
            linewidth=2.2, label='Pred ego traj')

    #* 高亮当前视频帧对应的未来时间点：frame_id=0对应未来第1帧。
    gt_k = min(frame_id + 1, len(gt_traj) - 1)
    pred_k = min(frame_id + 1, len(pred_traj) - 1)
    ax.scatter([gt_traj[gt_k, 0]], [gt_traj[gt_k, 1]], c='#00aa00',
               s=50, edgecolors='black', zorder=5)
    ax.scatter([pred_traj[pred_k, 0]], [pred_traj[pred_k, 1]], c='#ff6600',
               s=50, edgecolors='black', zorder=5)

    ax.scatter([0], [0], marker='^', s=90, c='#0066ff', edgecolors='black',
               zorder=6, label='current ego')
    ax.arrow(0, 0, 3, 0, width=0.08, head_width=0.7, color='#0066ff', zorder=6)

    ax.set_title(title)
    ax.set_xlabel('ego x / forward (m)')
    ax.set_ylabel('ego y / left (m)')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.3, alpha=0.4)


def render_compare_frame(gt_occ, pred_occ, gt_traj, pred_traj, frame_id, scene_text, args):
    """生成一张GT/Pred双栏BEV对比图并返回RGB数组。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    time_s = 0.5 * (frame_id + 1)
    draw_bev(axes[0], gt_occ, f'GT occupancy | future {frame_id + 1} ({time_s:.1f}s)',
             gt_traj, pred_traj, frame_id, args)
    draw_bev(axes[1], pred_occ, f'Pred occupancy | future {frame_id + 1} ({time_s:.1f}s)',
             gt_traj, pred_traj, frame_id, args)
    axes[1].legend(loc='upper right')
    fig.suptitle(scene_text)
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return rgb


def build_model_and_loader(cfg, args, logger):
    """按visualize_demo.py流程构建模型、加载checkpoint并创建验证集dataloader。"""
    import model  # noqa: F401  # 注册自定义模型到MODELS
    from dataset import get_dataloader

    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
    my_model = my_model.cuda()
    raw_model = my_model

    resume_from = ''
    latest = osp.join(args.work_dir, 'latest.pth')
    if osp.exists(latest):
        resume_from = latest
    if args.resume_from:
        resume_from = args.resume_from
    logger.info('resume from: ' + resume_from)
    if resume_from and osp.exists(resume_from):
        ckpt = torch.load(resume_from, map_location='cpu')
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        logger.info(f'successfully resumed from epoch {ckpt.get("epoch", "unknown")}')
    elif cfg.get('load_from', None):
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        logger.info('loaded from cfg.load_from')
    else:
        logger.warning('未加载checkpoint，将使用随机初始化权重，可视化结果通常无意义。')

    _, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False)
    return my_model.eval(), val_dataset_loader


def main():
    args = parse_args()
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    out_root = osp.join(args.work_dir, args.dir_name, timestamp)
    os.makedirs(out_root, exist_ok=True)
    logger = MMLogger('occworld_bev_vis', log_file=osp.join(out_root, 'visualize_autoreg_bev.log'))
    MMLogger._instance_dict['occworld_bev_vis'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    my_model, val_dataset_loader = build_model_and_loader(cfg, args, logger)
    os.environ['eval'] = 'true'

    with torch.no_grad():
        for i_iter_val, (input_occs, target_occs, metas) in enumerate(val_dataset_loader):
            if i_iter_val not in args.scene_idx:
                continue
            if i_iter_val > max(args.scene_idx):
                break

            input_occs = input_occs.cuda()
            result = my_model.forward_autoreg_with_pose(
                x=input_occs,
                metas=metas,
                start_frame=cfg.get('start_frame', 0),
                mid_frame=cfg.get('mid_frame', 5),
                end_frame=cfg.get('end_frame', 11))

            #* target_occs是未来GT Occupancy；sem_pred是自回归预测得到的未来Occupancy。
            gt_occs = to_numpy(result['target_occs'][0])
            pred_occs = to_numpy(result['sem_pred'][0])
            n_frames = min(len(gt_occs), len(pred_occs))

            #* gt_poses_ / poses_均为未来逐步位移(dx,dy)，这里转成从当前帧出发的累计轨迹。
            gt_traj = cumulative_trajectory(result['gt_poses_'][0])
            pred_traj = cumulative_trajectory(to_numpy(result['poses_'][0]))

            sample_dir = osp.join(out_root, f'sample_{i_iter_val:04d}')
            frames_dir = osp.join(sample_dir, 'frames')
            os.makedirs(sample_dir, exist_ok=True)
            if args.save_frames:
                os.makedirs(frames_dir, exist_ok=True)
            video_path = osp.join(sample_dir, 'autoreg_bev.mp4')

            scene_name = metas[0].get('scene_name', 'unknown') if isinstance(metas, list) else 'unknown'
            window_start = metas[0].get('window_start', 'unknown') if isinstance(metas, list) else 'unknown'
            scene_text = (f'val index={i_iter_val} | scene={scene_name} | window_start={window_start} | '
                          f'future_frames={n_frames}')

            writer = None
            try:
                for frame_id in range(n_frames):
                    rgb = render_compare_frame(
                        gt_occs[frame_id], pred_occs[frame_id],
                        gt_traj, pred_traj, frame_id, scene_text, args)
                    if writer is None:
                        height, width = rgb.shape[:2]
                        writer = cv2.VideoWriter(
                            video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f'无法创建视频：{video_path}')
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if args.save_frames:
                        cv2.imwrite(osp.join(frames_dir, f'{frame_id:03d}.png'),
                                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            finally:
                if writer is not None:
                    writer.release()

            logger.info(f'[{i_iter_val}] video saved: {video_path}')
            logger.info(f'[{i_iter_val}] gt_traj={gt_traj.tolist()}')
            logger.info(f'[{i_iter_val}] pred_traj={pred_traj.tolist()}')
            print(f'Video saved: {video_path}')


if __name__ == '__main__':
    main()
