import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmengine.registry import MODELS
try:
    from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
except ImportError:
    from timm.scheduler import CosineLRScheduler

from utils.load_save_util import revise_ckpt, revise_ckpt_1
import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", cfg.get("port", 29500))
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'eval_stp3_{cfg.start_frame}_{cfg.mid_frame}_{cfg.end_frame}_{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader, get_nuScenes_label_name
    from loss import OPENOCC_LOSS
    from utils.metric_util import MeanIoU, multi_step_MeanIou
    from utils.freeze_model import freeze_model

    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if cfg.get('freeze_dict', False):
        logger.info(f'Freezing model according to freeze_dict:{cfg.freeze_dict}')
        freeze_model(my_model, cfg.freeze_dict)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params after freezed: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume)

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config)
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False)

    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0
    best_val_iou = [0]*cfg.get('return_len_', 10)
    best_val_miou = [0]*cfg.get('return_len_', 10)

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']

        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if cfg.get('revise_ckpt', False):
            if cfg.revise_ckpt == 1:
                print('revise_ckpt')
                print(raw_model.load_state_dict(revise_ckpt(state_dict), strict=False))
            elif cfg.revise_ckpt == 2:
                print('revise_ckpt_1')
                print(raw_model.load_state_dict(revise_ckpt_1(state_dict), strict=False))
            elif cfg.revise_ckpt == 3:
                print('revise_ckpt_2')
                print(raw_model.vae.load_state_dict(state_dict, strict=False))
        else:
            print(raw_model.load_state_dict(state_dict, strict=False))

    # training
    print_freq = cfg.print_freq
    first_run = True
    grad_norm = 0

    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    #*==================== Occupancy多步预测指标统计器 ====================#
    #* multi_step_MeanIou会按未来每个预测时间步分别维护混淆矩阵，而不是把所有未来帧混在一起统计。
    #* sem统计语义mIoU；vox先把语义occupancy转成二值占用/空闲，再统计占用IoU。
    #* times=eval_length表示评估未来eval_length个预测帧，最终_after_epoch()返回长度为eval_length的指标列表。
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('eval_length'))
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('eval_length'))

    my_model.eval()
    os.environ['eval'] = 'true'
    val_loss_list = []
    CalMeanIou_sem.reset()
    CalMeanIou_vox.reset()
    #*==================== ST-P3风格规划指标累加器 ====================#
    #* autoreg_for_stp3_metric内部会在约1s/2s/3s三个时间点分别计算：
    #*   plan_L2：预测自车轨迹与GT自车轨迹的L2误差；
    #*   plan_obj_col：基于栅格/占用的碰撞率；
    #*   plan_obj_box_col：基于3D bbox投影的碰撞率。
    #* 非single版本表示从当前时刻t0累计到对应时间点的区间指标：
    #*   1s = 0~1s，2s = 0~2s，3s = 0~3s，而不是1~2s、2~3s这种分段指标。
    #* *_single是对应时间点的单点指标：只看约1s/2s/3s那一帧，不看此前整段轨迹。
    #* 后面会先对验证集样本求平均，再对1s/2s/3s求总平均。
    metric_stp3 = {
            'plan_L2_1s':0,  # 0~1s整段自车轨迹L2误差，后续累加所有验证样本后取平均
            'plan_L2_2s':0,  # 0~2s整段自车轨迹L2误差
            'plan_L2_3s':0,  # 0~3s整段自车轨迹L2误差
            'plan_obj_col_1s':0,  # 0~1s整段轨迹基于occupancy栅格的碰撞率/碰撞指标
            'plan_obj_col_2s':0,  # 0~2s整段轨迹基于occupancy栅格的碰撞率/碰撞指标
            'plan_obj_col_3s':0,  # 0~3s整段轨迹基于occupancy栅格的碰撞率/碰撞指标
            'plan_obj_box_col_1s':0,  # 0~1s整段轨迹基于GT 3D bbox的碰撞率/碰撞指标
            'plan_obj_box_col_2s':0,  # 0~2s整段轨迹基于GT 3D bbox的碰撞率/碰撞指标
            'plan_obj_box_col_3s':0,  # 0~3s整段轨迹基于GT 3D bbox的碰撞率/碰撞指标
            'plan_L2_1s_single':0,  # 只看约1s单个时刻的L2误差，不累计0~1s整段
            'plan_L2_2s_single':0,  # 只看约2s单个时刻的L2误差，不累计0~2s整段
            'plan_L2_3s_single':0,  # 只看约3s单个时刻的L2误差，不累计0~3s整段
            'plan_obj_col_1s_single':0,  # 只看约1s单个时刻的occupancy碰撞指标
            'plan_obj_col_2s_single':0,  # 只看约2s单个时刻的occupancy碰撞指标
            'plan_obj_col_3s_single':0,  # 只看约3s单个时刻的occupancy碰撞指标
            'plan_obj_box_col_1s_single':0,  # 只看约1s单个时刻的bbox碰撞指标
            'plan_obj_box_col_2s_single':0,  # 只看约2s单个时刻的bbox碰撞指标
            'plan_obj_box_col_3s_single':0,  # 只看约3s单个时刻的bbox碰撞指标
    }
    time_used = {
        'encode':0,  # VQ-VAE Encoder/量化等前处理耗时累计
        'mid':0,  # 自回归开始前，中间元数据处理/pose编码等耗时累计
        'autoreg':0,  # 逐帧自回归预测未来场景和自车运动的耗时累计
        'total':0,  # 单个batch完整推理流程总耗时累计
        'per_frame':0,  # 折算到每个预测帧的平均耗时累计，用于估算FPS
    }
    with torch.no_grad():
        plan_loss = 0
        for i_iter_val, (input_occs, target_occs, metas) in enumerate(val_dataset_loader):

            input_occs = input_occs.cuda()
            target_occs = target_occs.cuda()
            data_time_e = time.time()
            if cfg.get('eval_with_pose', False):
                # *==================================================#
                #* 自回归推理：以历史帧为条件逐步生成未来场景和自车轨迹。
                #* 输出中同时包含：
                #*   sem_pred / iou_pred：未来每个时间步的occupancy预测，用于逐时间步IoU/mIoU；
                #*   metric_stp3：1s/2s/3s处的规划L2与碰撞指标。
                if not distributed:
                    result_dict = my_model.autoreg_for_stp3_metric(
                        x=input_occs, metas=metas,
                        start_frame=cfg.get('start_frame', 0),
                        mid_frame=cfg.get('mid_frame', 6),
                        end_frame=cfg.get('end_frame', 12))
                else:
                    result_dict = my_model.module.autoreg_for_stp3_metric(
                        x=input_occs, metas=metas,
                        start_frame=cfg.get('start_frame', 0),
                        mid_frame=cfg.get('mid_frame', 6),
                        end_frame=cfg.get('end_frame', 12))
            else:
                raise NotImplementedError

            for key in metric_stp3.keys():
                metric_stp3[key] += result_dict['metric_stp3'][key]
            for key in time_used.keys():
                time_used[key] += result_dict['time'][key]
            loss_input = {
                'inputs': input_occs,
                'target_occs': target_occs,
                # 'metas': metas
            }
            for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                loss_input.update({
                    loss_input_key: result_dict[loss_input_val]
                })
            loss, loss_dict = loss_func(loss_input)
            plan_loss += loss_dict.get('PlanRegLoss', 0)
            plan_loss += loss_dict.get('PlanRegLossLidar', 0)
            if result_dict.get('target_occs', None) is not None:
                target_occs = result_dict['target_occs']
            target_occs_iou = deepcopy(target_occs)
            #* 二值占用IoU只关心“是否被占用”：nuScenes/Occ3D中17通常表示free，其余语义类视为occupied。
            target_occs_iou[target_occs_iou != 17] = 1
            target_occs_iou[target_occs_iou == 17] = 0
            #* ===========================================================#
            #* Occupancy评估：_after_step不会立刻汇总成一个数，而是按未来时间维逐帧累加统计量。
            #* 因此最终val_iou[k] / val_miou[k]分别表示第k个未来预测帧的IoU / mIoU。
            CalMeanIou_sem._after_step(result_dict['sem_pred'], target_occs)
            CalMeanIou_vox._after_step(result_dict['iou_pred'], target_occs_iou)
            val_loss_list.append(loss.detach().cpu().numpy())
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Epoch %d Iter %5d/%5d: Loss: %.3f (%.3f)'%(
                    epoch, i_iter_val,len(val_dataset_loader), loss.item(), np.mean(val_loss_list)))
                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)

    #* ===========================================================#
    #* 规划指标先对整个验证集求平均，得到1s/2s/3s三个时间点各自的平均L2/碰撞指标。
    metric_stp3 = {key:metric_stp3[key]/len(val_dataset_loader) for key in metric_stp3.keys()}
    time_used = {key:time_used[key]/len(val_dataset_loader) for key in time_used.keys()}
    # reduce for distributed
    if distributed:
        plan_loss = torch.tensor(plan_loss, dtype=torch.float64).cuda()
        dist.all_reduce(plan_loss)
        plan_loss /= world_size
        metric_stp3 = {key:torch.tensor(metric_stp3[key],dtype=torch.float64).cuda() for key in metric_stp3.keys()}
        for key in metric_stp3.keys():
            dist.all_reduce(metric_stp3[key])
            metric_stp3[key] /= world_size
        time_used = {key:torch.tensor(time_used[key],dtype=torch.float64).cuda() for key in time_used.keys()}
        for key in time_used.keys():
            dist.all_reduce(time_used[key])
            time_used[key] /= world_size

    #* ===========================================================#
    #* 正式汇总规划指标时，再把1s、2s、3s三个时间点等权平均，得到整体规划分数。
    metric_stp3.update(avg_l2=(metric_stp3['plan_L2_1s']+metric_stp3['plan_L2_2s']+metric_stp3['plan_L2_3s'])/3)
    metric_stp3.update(avg_obj_col=(metric_stp3['plan_obj_col_1s']+metric_stp3['plan_obj_col_2s']+metric_stp3['plan_obj_col_3s'])/3)
    metric_stp3.update(avg_obj_box_col=(metric_stp3['plan_obj_box_col_1s']+metric_stp3['plan_obj_box_col_2s']+metric_stp3['plan_obj_box_col_3s'])/3)
    metric_stp3.update(avg_obj_box_col_single=(metric_stp3['plan_obj_box_col_1s_single']+metric_stp3['plan_obj_box_col_2s_single']+metric_stp3['plan_obj_box_col_3s_single'])/3)
    metric_stp3.update(avg_obj_col_single=(metric_stp3['plan_obj_col_1s_single']+metric_stp3['plan_obj_col_2s_single']+metric_stp3['plan_obj_col_3s_single'])/3)
    metric_stp3.update(avg_l2_single=(metric_stp3['plan_L2_1s_single']+metric_stp3['plan_L2_2s_single']+metric_stp3['plan_L2_3s_single'])/3)
    for key in metric_stp3.keys():
        metric_stp3[key] = metric_stp3[key].item()
        logger.info(f'{key} is {metric_stp3[key]}')
    #logger.info(f'metric_stp3 is {metric_stp3}')
    logger.info(f'time_used is {time_used}')
    logger.info(f'FPS is {1/time_used["per_frame"]}')

    #* ===========================================================#
    #* Occupancy指标在这里完成跨验证集汇总；返回list中每一项对应一个未来预测时间步。
    val_miou, _ = CalMeanIou_sem._after_epoch()
    val_iou, _ = CalMeanIou_vox._after_epoch()
    logger.info(f'PlanRegLoss is {plan_loss/len(val_dataset_loader)}')
    del target_occs, input_occs

    #best_val_iou = [max(best_val_iou[i], val_iou[i]) for i in range(len(best_val_iou))]
    #best_val_miou = [max(best_val_miou[i], val_miou[i]) for i in range(len(best_val_miou))]

    logger.info(f'Current val iou is {val_iou}')
    logger.info(f'Current val miou is {val_miou}')
    #* ===========================================================#
    #* OccWorld/ST-P3评估通常关注约1s、2s、3s三个预测时刻。
    #* nuScenes关键帧间隔约0.5s，因此未来帧索引1/3/5分别近似对应1s/2s/3s。
    logger.info(f'avg val iou is {(val_iou[1]+val_iou[3]+val_iou[5])/3}')
    logger.info(f'avg val miou is {(val_miou[1]+val_miou[3]+val_miou[5])/3}')
    #logger.info(f'Current val iou is {val_iou} while the best val iou is {best_val_iou}')
    #logger.info(f'Current val miou is {val_miou} while the best val miou is {best_val_miou}')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
