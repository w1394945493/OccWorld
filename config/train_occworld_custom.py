grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 50
return_len_ = 15
return_len_train = 15 # # *  二阶段训练return_len = return_len_train + 1 = 16 每个batch_sample取连续16帧，第16帧不是历史输入，作为GT监督

batch_size = 2  # 每张GPU每个iteration加载的时序样本数；多卡全局BS=batch_size×GPU数
num_workers = 2  # 每个DataLoader进程使用的数据读取子进程数；训练集和验证集共用该配置
save_every_epochs = 1  # 每训练1个epoch保存一次checkpoint，并更新latest.pth软链接
max_keep_ckpts = 1  # 最多保留最新5个epoch_N.pth；设为0或负数时不自动删除旧权重
eval_every_epochs = 1  # 每训练1个epoch在验证集上执行一次评估
# eval_every_epochs = 1
# save_every_epochs = 1

load_from = "/c20250502/wangyushen/Outputs/occworld/vqvae/train/epoch_200.pth"  # recommend selecting the best vqvae model
port = 25096
revise_ckpt = 3

multisteplr = False
multisteplr_config = dict(
    decay_t = [87 * 500],
    decay_rate = 0.1,
    warmup_t = warmup_iters,
    warmup_lr_init = 1e-6,
    t_in_epochs = False)

freeze_dict = dict(
    vae = True,
    transformer = False,
    pose_encoder = False,
    pose_decoder = False,
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
    return_len=return_len_train + 1,
    offset=0,
    imageset="/c20250502/wangyushen/Datasets/NuScenes/method/occworld/nuscenes_infos_train_temporal_v3_scene.pkl",
)

val_dataset_config = dict(
    type="nuScenesSceneDatasetLidar",
    data_path=data_path,
    occ_path=occ_path,
    return_len=return_len_ + 1,
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
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

val_loader = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='CeLoss',
            weight=1.0,
            input_dict={
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'}),
        dict(
            type='PlanRegLossLidar',
            weight=0.1,
            loss_type='l2',
            num_modes=3,
            input_dict={
                'rel_pose': 'rel_pose',
                'metas': 'metas'})
    ]
)


loss_input_convertion = dict(
    ce_inputs = 'ce_inputs',
    ce_labels = 'ce_labels',
    rel_pose='pose_decoded',
    metas ='output_metas',
)

base_channel = 64
_dim_ = 16
expansion = 8
n_e_ = 512
model = dict(
    type = 'TransVQVAE',
    num_frames=return_len_,
    delta_input=False,
    offset=1,
    vae = dict(
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
            sane_index_shape=True,
            n_e = n_e_,
            e_dim = base_channel * 2,
            beta = 1.,
            z_channels = base_channel * 2,
            use_voxel=False)),

    transformer=dict(
        type = 'PlanUAutoRegTransformer',
        num_tokens=1,
        num_frames=return_len_,
        num_layers=2,
        img_shape=(base_channel*2,50,50),
        pose_shape=(1,base_channel*2),
        pose_attn_layers=2,
        pose_output_channel=base_channel*2,
        tpe_dim=base_channel*2,
        channels=(base_channel*2, base_channel*4, base_channel*8),
        temporal_attn_layers=6,
        output_channel=n_e_,
        learnable_queries=False
    ),
    pose_encoder=dict(
        type = 'PoseEncoder',
        in_channels=5,
        out_channels=base_channel*2,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
    pose_decoder=dict(
        type = 'PoseDecoder',
        in_channels=base_channel*2,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
)


shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"
