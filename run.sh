
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
  --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/OccWorld/config/train_occworld_custom.py \
  --work-dir out/occworld



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
