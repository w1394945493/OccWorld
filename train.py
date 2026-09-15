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


def remove_epoch_checkpoints(work_dir, max_keep_ckpts, logger):
    """只保留最新的若干个epoch checkpoint；不处理latest/iter/best等其他权重。"""
    if max_keep_ckpts is None or max_keep_ckpts <= 0:
        return  # None或非正数表示不限制保存数量

    epoch_ckpts = []
    for filename in os.listdir(work_dir):
        if filename.startswith('epoch_') and filename.endswith('.pth'):
            epoch_str = filename[len('epoch_'):-len('.pth')]
            if epoch_str.isdigit():
                epoch_ckpts.append((int(epoch_str), osp.join(work_dir, filename)))

    epoch_ckpts.sort(key=lambda item: item[0])
    for _, ckpt_path in epoch_ckpts[:-max_keep_ckpts]:
        os.remove(ckpt_path)
        logger.info(f'Removed old checkpoint: {ckpt_path}')


def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    #*==================== 根据可见GPU数量初始化单卡或DDP多卡训练 ====================#
    # args.gpus由程序入口的torch.cuda.device_count()自动得到，而不是命令行参数。
    # 可在启动前通过CUDA_VISIBLE_DEVICES限制可见GPU；例如CUDA_VISIBLE_DEVICES=2,3会启动2个进程。
    if args.gpus > 1:
        distributed = True  # 可见GPU超过1张时自动启用DistributedDataParallel
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")  # 默认在本机建立进程通信
        port = os.environ.get("MASTER_PORT", cfg.get("port", 29500))  # DDP通信端口
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # 节点数量；普通单机训练默认为1
        rank = int(os.environ.get("RANK", 0))  # 当前节点编号；普通单机训练默认为0
        gpus = torch.cuda.device_count()  # 当前节点可见GPU数，即每节点启动的训练进程数
        print(f"tcp://{ip}:{port}")
        #* 每个进程绑定一张GPU；全局进程数=节点数×每节点可见GPU数。
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()  # 取得参与DDP训练的全局进程总数
        cfg.gpu_ids = range(world_size)  # 记录全局GPU/进程编号范围
        torch.cuda.set_device(local_rank)  # 当前子进程只在对应的逻辑GPU上执行

        # 非0号进程关闭普通print，避免多卡时重复输出；训练日志主要由local_rank=0记录。
        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False  # 仅有1张可见GPU时使用单进程；若为0，后续.cuda()会因无GPU而报错
        world_size = 1

    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
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

    #*==================== 自动断点续训与预训练权重加载 ====================#
    #* resume会恢复模型、优化器、学习率调度器和训练进度；load_from通常只加载模型参数。
    epoch = 0  # 已完成的epoch数；恢复后while循环从该值继续
    global_iter = 0  # 已执行的全局优化步数，用于恢复iteration级学习率进度
    last_iter = 0  # 当前epoch内已执行的iteration数，仅--iter-resume场景使用
    best_val_iou = [0]*cfg.get('return_len_', 10)
    best_val_miou = [0]*cfg.get('return_len_', 10)

    #* 自动恢复只检查work_dir/latest.pth，不会遍历目录寻找编号最大的epoch_N.pth。
    # latest.pth由每次保存checkpoint时更新，正常情况下始终指向最近一次保存的权重。
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')  # 找到latest.pth后自动启用断点续训
    #* 命令行--resume-from优先级更高，可显式指定任意epoch_N.pth或iter.pth。
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        #* 先加载到CPU以避免读取checkpoint时额外占用GPU显存，再分别恢复完整训练状态。
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))  # 恢复模型参数
        optimizer.load_state_dict(ckpt['optimizer'])  # 恢复AdamW动量等优化器状态
        scheduler.load_state_dict(ckpt['scheduler'])  # 恢复学习率调度进度
        epoch = ckpt['epoch']  # 恢复已完成epoch数；如为80，后续继续训练至max_epochs
        global_iter = ckpt['global_iter']  # 恢复全局iteration计数
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0  # epoch权重没有该字段时从本epoch第0步开始
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']

        #* 自定义Sampler支持跳过当前epoch内已经训练过的数据，实现iteration级续训。
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    #* 仅当没有可用resume checkpoint时才处理load_from；该分支不会恢复优化器和训练进度。
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
    #* 两个评估器都按时间位置分别累计整个验证集的混淆矩阵；times=return_len_时最终返回等长列表。
    # 例如return_len_=10，列表第i项表示所有验证窗口中第i帧汇总后的指标，不是第i个语义类别。
    # sem：分别计算各有效语义类别IoU后取类别均值，得到每个时间位置的语义mIoU。
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('return_len_', 10))
    # vox：把所有非空语义合并为occupied类，得到每个时间位置的二值Occupancy IoU。
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('return_len_', 10))

    # logger.info('compiling model')
    # my_model = torch.compile(my_model)
    # logger.info('done compile model')
    best_plan_loss = 100000
    while epoch < max_num_epochs:

        my_model.train()
        os.environ['eval'] = 'false'
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, (input_occs, target_occs, metas) in enumerate(train_dataset_loader):
            if first_run:
                i_iter = i_iter + last_iter

            input_occs = input_occs.cuda() # (1 10 200 20 16)
            target_occs = target_occs.cuda() # (1 10 200 200 16)
            data_time_e = time.time()

            result_dict = my_model(x=input_occs, metas=metas)

            loss_input = {
                'inputs': input_occs,
                'target_occs': target_occs,
                # 'metas': metas
            }

            for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                loss_input.update({
                    loss_input_key: result_dict[loss_input_val]})
            loss, loss_dict = loss_func(loss_input)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            optimizer.step()

            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and local_rank == 0:
                lr = optimizer.param_groups[0]['lr']
                #* 记录当前训练进程从启动以来在本张GPU上的峰值Tensor显存，便于判断是否接近OOM。
                # 该值不包含PyTorch预留但尚未被Tensor使用的缓存，也不包含同一GPU上的其他进程。
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Byte转换为GB
                #* MMLogger会将本条训练信息同时打印到终端，并写入work_dir下的时间戳.log文件。
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, peak_mem: %.2f GB, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader),
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    peak_mem_gb, time_e - time_s, data_time_e - data_time_s))

                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
                loss_list = []
            data_time_s = time.time()
            time_s = time.time()

            if args.iter_resume:
                if (i_iter + 1) % 50 == 0 and local_rank == 0:
                    dict_to_save = {
                        'state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'last_iter': i_iter + 1,
                    }
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), 'iter.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest.pth')
                    symlink(save_file_name, dst_file)  #* latest.pth改为指向最新iter权重，供下次自动续训
                    logger.info(f'iter ckpt {i_iter + 1} saved!')

        #*==================== Epoch级Checkpoint保存与旧权重轮转 ====================#
        if local_rank == 0 and (epoch + 1) % cfg.get('save_every_epochs', 1) == 0:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            #* 每次保存后更新latest.pth；自动续训实际读取该链接，而不是搜索最大的epoch编号。
            symlink(save_file_name, dst_file)
            #* 保存新权重后，仅轮转清理最旧的epoch_N.pth；latest.pth及其他类型权重不受影响。
            remove_epoch_checkpoints(
                args.work_dir, cfg.get('max_keep_ckpts', 0), logger)

        epoch += 1
        first_run = False

        # ========================================================#
        # eval
        if epoch % cfg.get('eval_every_epochs', 1) != 0:
            continue
        my_model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []
        CalMeanIou_sem.reset()
        CalMeanIou_vox.reset()
        plan_loss = 0

        with torch.no_grad():
            for i_iter_val, (input_occs, target_occs, metas) in enumerate(val_dataset_loader):

                input_occs = input_occs.cuda()
                target_occs = target_occs.cuda()
                data_time_e = time.time()

                result_dict = my_model(x=input_occs, metas=metas)

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
                #* 为二值Occupancy IoU构造GT：类别17为空体素，其余语义类别统一视为occupied。
                target_occs_iou = deepcopy(target_occs)
                target_occs_iou[target_occs_iou != 17] = 1
                target_occs_iou[target_occs_iou == 17] = 0

                #* 当前batch按时间位置累加统计量；不是先计算每个样本IoU后再做简单平均。
                CalMeanIou_sem._after_step(result_dict['sem_pred'], target_occs)
                CalMeanIou_vox._after_step(result_dict['iou_pred'], target_occs_iou)
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    detailed_loss = ', '.join(detailed_loss)
                    logger.info(detailed_loss)
        #* 汇总整个验证集：val_miou/val_iou均为长度return_len_的列表，每项对应一个时间位置。
        # VQ-VAE的offset=0时，这些位置是连续窗口内各帧的重建指标，并非不同未来预测步。
        val_miou, _ = CalMeanIou_sem._after_epoch()  # 每个时间位置的语义类别平均IoU
        val_iou, _ = CalMeanIou_vox._after_epoch()  # 每个时间位置的二值占用IoU

        del target_occs, input_occs
        plan_loss = plan_loss/len(val_dataset_loader)
        if plan_loss < best_plan_loss:
            best_plan_loss = plan_loss
        logger.info(f'PlanRegLoss is {plan_loss} while the best plan loss is {best_plan_loss}')
        #logger.info(f'PlanRegLoss is {plan_loss/len(val_dataset_loader)}')
        #* 每个时间位置独立维护历史最优值，因此列表中的最佳项可能分别来自不同epoch。
        best_val_iou = [max(best_val_iou[i], val_iou[i]) for i in range(len(best_val_iou))]
        best_val_miou = [max(best_val_miou[i], val_miou[i]) for i in range(len(best_val_miou))]
        #logger.info(f'PlanRegLoss is {plan_loss/len(val_dataset_loader)}')
        logger.info(f'Current val iou is {val_iou} while the best val iou is {best_val_iou}')
        logger.info(f'Current val miou is {val_miou} while the best val miou is {best_val_miou}')
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

    #* 自动统计当前进程可见的GPU数量；脚本没有单独的--gpus数量参数。
    # CUDA_VISIBLE_DEVICES=0时ngpus=1；CUDA_VISIBLE_DEVICES=2,3时ngpus=2且内部逻辑编号为0、1。
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus  # 将自动检测结果传入main，用于选择单卡或DDP分支
    print(args)

    # *=================================================================#
    #* 多于1张可见GPU时，每张卡启动一个main子进程；否则直接在当前进程执行main。
    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
