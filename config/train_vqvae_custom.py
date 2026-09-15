grad_max_norm = 35
print_freq = 10  # 每训练10个iteration打印一次loss、学习率、梯度范数和耗时等日志（第0步也会打印）
max_epochs = 200  # 最大训练轮数；train.py会完整遍历训练集200次
warmup_iters = 200  # 学习率预热步数；前200个优化iteration内从warmup_lr_init逐渐升至基础学习率
return_len_ = 10  # 每个数据样本包含的连续Occupancy帧数；表示时序长度F，不是batch size # * 一阶段训练return_len_=10, offset = 0: 每个batch sample取连续10帧occupancy

batch_size = 2  # 每张GPU每个iteration加载的时序样本数；多卡全局BS=batch_size×GPU数
num_workers = 2  # 每个DataLoader进程使用的数据读取子进程数；训练集和验证集共用该配置
save_every_epochs = 1  # 每训练1个epoch保存一次checkpoint，并更新latest.pth软链接
max_keep_ckpts = 1  # 最多保留最新5个epoch_N.pth；设为0或负数时不自动删除旧权重
eval_every_epochs = 1  # 每训练1个epoch在验证集上执行一次评估


multisteplr = False
multisteplr_config = dict(
    decay_t = [87 * 500],
    decay_rate = 0.1,
    warmup_t = warmup_iters,
    warmup_lr_init = 1e-6,
    t_in_epochs = False
)


optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
    ),
)

data_path = "/c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval/"
occ_path = "/c20250502/wangyushen/Datasets/occ3d_nuscenes/"


train_dataset_config = dict(
    type="nuScenesSceneDatasetLidar",
    data_path=data_path,
    occ_path=occ_path,
    return_len=return_len_,
    offset=0,
    imageset="/c20250502/wangyushen/Datasets/NuScenes/method/occworld/nuscenes_infos_train_temporal_v3_scene.pkl",
)

val_dataset_config = dict(
    type="nuScenesSceneDatasetLidar",
    data_path=data_path,
    occ_path=occ_path,
    return_len=return_len_,
    offset=0,
    imageset="/c20250502/wangyushen/Datasets/NuScenes/method/occworld/nuscenes_infos_val_temporal_v3_scene.pkl",
)

train_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes',
    phase='train',
)

val_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes',
    phase='val',
)

train_loader = dict(
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
)

val_loader = dict(
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
)


loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='ReconLoss',
            weight=10.0,
            ignore_label=-100,
            use_weight=False,
            cls_weight=None,
            input_dict={
                'logits': 'logits',
                'labels': 'inputs'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            input_dict={
                'logits': 'logits',
                'labels': 'inputs'}),
        dict(
            type='VQVAEEmbedLoss',
            weight=1.0),
        ])

loss_input_convertion = dict(
    logits='logits',
    embed_loss='embed_loss'
)


load_from = ''

_dim_ = 16
expansion = 8
base_channel = 64
n_e_ = 512
model = dict(
    type = 'VAERes2D',
    encoder_cfg=dict(
        type='Encoder2D',
        ch = base_channel,
        out_ch = base_channel,
        ch_mult = (1,2,4),
        num_res_blocks = 2,
        attn_resolutions = (50,),
        dropout = 0.0,
        resamp_with_conv = True,
        in_channels = _dim_ * expansion,
        resolution = 200,
        z_channels = base_channel * 2,
        double_z = False,
    ),
    decoder_cfg=dict(
        type='Decoder2D',
        ch = base_channel,
        out_ch = _dim_ * expansion,
        ch_mult = (1,2,4),
        num_res_blocks = 2,
        attn_resolutions = (50,),
        dropout = 0.0,
        resamp_with_conv = True,
        in_channels = _dim_ * expansion,
        resolution = 200,
        z_channels = base_channel * 2,
        give_pre_end = False
    ),
    num_classes=18,
    expansion=expansion,
    vqvae_cfg=dict(
        type='VectorQuantizer',
        n_e = n_e_,
        e_dim = base_channel * 2,
        beta = 1.,
        z_channels = base_channel * 2,
        use_voxel=False))

shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"
