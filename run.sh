

python tools/occworld_nuscenes_converter.py \
  --root-path /c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval \
  --canbus /c20250502/wangyushen/Datasets/NuScenes \
  --occ3d-root /c20250502/wangyushen/Datasets/occ3d_nuscenes/gts \
  --out-dir /c20250502/wangyushen/Datasets/NuScenes/method/occworld \
  --extra-tag nuscenes \
  --version v1.0 \
  --max-sweeps 10 \
  --min-scene-frames 16


python eval_metric_stp3.py \
  --py-config config/occworld.py \
  --work-dir out/occworld