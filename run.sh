
#  数据集meta pkl生成
python tools/occworld_nuscenes_converter.py \
  --root-path /c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval \
  --canbus /c20250502/wangyushen/Datasets/NuScenes \
  --occ3d-root /c20250502/wangyushen/Datasets/occ3d_nuscenes/gts \
  --out-dir /c20250502/wangyushen/Datasets/NuScenes/method/occworld \
  --extra-tag nuscenes \
  --version v1.0 \
  --max-sweeps 10 \
  --min-scene-frames 16

# 推理 四维Occ+自车轨迹预测
python eval_metric_stp3.py \
  --py-config config/occworld.py \
  --work-dir out/occworld

# ========================================================#
# stage1
# 训练 VQ-VAE
python train.py \
  --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/OccWorld/config/train_vqvae_custom.py \
  --work-dir out/vqvae


# 可视化原始数据集
python tools/visualize_dataset_sequence_bev.py \
  --py-config config/train_vqvae_custom.py \
  --split val \
  --seed 42 \
  --motion turn \
  --output out/dataset_check/val_random.mp4 \
  --fps 2 \
  --save-frames

# ========================================================#
# stage2 完整训练
python train.py \
  --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/OccWorld/config/train_occworld_debug.py \
  --work-dir out/occworld

# 可视化

python3 /vepfs-mlp2/c20250502/haoce/wangyushen/OccWorld/tools/visualize_autoreg_bev.py \
  --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/OccWorld/config/occworld_custom.py \
  --resume-from /c20250502/wangyushen/Outputs/occworld/occworld/train/latest.pth \
  --work-dir out/occworld_custom \
  --scene-idx 500 1000 \
  --fps 2 \
  --save-frames

# ========================================================#
# stage1: VQ-VAE
cd /vepfs-mlp2/c20250502/haoce/wangyushen/OccWorld/
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wangyushentemp
bash sh/train_stage1.sh

# stage2: OccWorld
cd /vepfs-mlp2/c20250502/haoce/wangyushen/OccWorld/
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wangyushentemp
bash sh/train_stage2.sh


# ===================================================================================#
# semantic kitti
# 构造 SemanticKITTI 最小 OccWorld 风格 pkl：
#   - 只需提供 SemanticKITTI dataset 根目录；
#   - dense occupancy 路径会按 FoundationSSC 规则自动推断为：
#       <data-root>/labels/<sequence>/<frame_id>_1_1.npy
#   - 从 sequence 00 的第100帧开始，连续保存100帧到同一个 scene(sequence-00) 中。
python3 tools/semantickitti_converter.py \
  --data-root /c20250502/wangyushen/Datasets/kitti/semantickitti/dataset \
  --sequence 00 \
  --frame-idx 100 \
  --num-frames 100 \
  --out-pkl out/semkitti/semantickitti_seq00_000100_100frames.pkl

# 离线 BEV 可视化上述 pkl：
#   - pkl metadata 已记录 data_root，因此这里无需再传 --data-root；
#   - 输出 out/semantickitti_vis/sequence-00/bev.mp4；
#   - 加 --save-frames 时，同时保存逐帧 png 到 frames/。
python3 tools/visualize_semantickitti_pkl_bev.py \
  --pkl out/semkitti/semantickitti_seq00_000100_100frames.pkl \
  --output-dir out/semantickitti_vis \
  --scene sequence-00 \
  --max-frames 100 \
  --fps 5 \
  --save-frames
