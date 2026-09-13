import os, numpy as np, pickle
from pyquaternion import Quaternion
from copy import deepcopy
from . import OPENOCC_DATASET
import torch

from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, Box3DMode


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar:
    def __init__(
            self, 
            data_path,
            occ_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            times=5,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        #! OccWorld 特有的 pkl 组织约定：data['infos'] 不是 VAD 的逐帧 list，而是
        #! {scene_name: [frame_info, ...]} 形式的 dict。键通常为 'scene-0001' 等
        #! nuScenes 场景名称，值是该场景内按时间排列的连续关键帧元数据列表。
        self.nusc_infos = data['infos']
        #! scene_names 保存上述 dict 的场景名称键；这些名称后面还会直接用于拼接
        #! Occ3D occupancy 标签目录，因此不能简单替换成 nuScenes scene token(UUID)。
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.occ_path = occ_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            #! OccWorld 特有的 occupancy 数据组织：
            #! <data_path>/<gts|tpv_dense|tpv_sparse>/<scene-xxxx>/<sample_token>/labels.npz。
            #! VAD 元数据只提供 sample token，不包含 labels.npz；转换 pkl 时必须确保
            #! dict 键 scene_name 与实际 Occ3D 标签中的 'scene-xxxx' 目录名完全一致。
            label_file = os.path.join(self.occ_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # input_occs = np.stack(occs, dtype=np.int64)
        input_occs = np.stack(occs).astype(np.int64, copy=False)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            #! output_dataset 当前被限制为 gts，用作未来 occupancy 的监督/评估目标。
            label_file = os.path.join(self.occ_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # output_occs = np.stack(occs, dtype=np.int64)
        output_occs = np.stack(occs).astype(np.int64, copy=False)
        metas = {}
        #! 注意：这里虽然命名为 scene_token，实际写入的是该场景第 5 个关键帧的
        #! sample token，并非 nuScenes 原生的 scene token；这是 OccWorld 当前代码的接口约定。
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        metas.update(self.get_meta_data(scene_name, idx))
        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))
        return input_occs[:self.return_len], output_occs[self.offset:], metas

    def get_meta_data(self, scene_name, idx):
        #! OccWorld 从连续窗口内每一帧的 VAD 风格未来轨迹中只取第 0 步，即该帧到
        #! 下一关键帧的 (dx, dy)，再将这些单步位移组织成世界模型使用的 rel_poses。
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2
            #! pose_mode 是 OccWorld 元数据额外要求的三模态驾驶/规划标签，通常为
            #! shape=(3,) 的 one-hot；VAD converter 不生成此键，只生成 gt_ego_fut_cmd。
            #! 使用 VAD pkl 转换时需显式补充，例如 pose_mode = gt_ego_fut_cmd，且应
            #! 确认右转/左转/直行三个 mode 的顺序与所用 OccWorld checkpoint 一致。
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {'rel_poses': xys, 'gt_mode': gt_modes}
    def get_image_info(self, scene_name, idx):
        #! OccWorld 的时序窗口以固定 T=6 选择参考帧：参考下标为窗口末帧下标减 6。
        #! 这是 OccWorld 的历史/未来帧布局约定，不是 VAD pkl 中自带的字段。
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t

            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            # cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            # focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))

        return input_dict
@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarTraverse(nuScenesSceneDatasetLidar):
    def __init__(
        self,
        data_path,
        occ_path,
        return_len,
        offset,
        imageset='train',
        nusc=None,
        times=1,
        test_mode=False,
        use_valid_flag=True,
        input_dataset='gts',
        output_dataset='gts',
    ):
        super().__init__(
            data_path,
            occ_path, 
            return_len,
            offset,
            imageset,
            nusc,
            times,
            test_mode,
            input_dataset,
            output_dataset,
        )
        self.scene_lens = [l - self.return_len - self.offset for l in self.scene_lens]
        self.use_valid_flag = use_valid_flag
        self.CLASSES = [
            'noise', 'animal' ,'human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
            'movable_object.barrier', 'movable_object.debris',
            'movable_object.pushable_pullable', 'movable_object.trafficcone',
            'static_object.bicycle_rack', 'vehicle.bicycle',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
            'vehicle.construction', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.motorcycle',
            'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
            'flat.other', 'flat.sidewalk', 'flat.terrain', 'flat.traffic_marking',
            'static.manmade', 'static.other', 'static.vegetation',
            'vehicle.ego'
        ]
        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

    def __len__(self):
        'Denotes the total number of samples'
        return sum(self.scene_lens)

    def __getitem__(self, index):
        for i, scene_len in enumerate(self.scene_lens):
            if index < scene_len:
                scene_name = self.scene_names[i]
                idx = index
                break
            else:
                index -= scene_len
        occs = []
        for i in range(self.return_len + self.offset): # self.return_len:12
            token = self.nusc_infos[scene_name][idx + i]['token']
            #! 与父类相同，scene_name 必须是 'scene-xxxx' 一类目录名；VAD pkl 的
            #! info['scene_token'] 需要借助 nuScenes scene 表映射成该名称后再分组。
            label_file = os.path.join(self.occ_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # input_occs = np.stack(occs, dtype=np.int64)
        input_occs = np.stack(occs).astype(np.int64, copy=False)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.occ_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        # output_occs = np.stack(occs, dtype=np.int64)
        output_occs = np.stack(occs).astype(np.int64, copy=False)
        metas = {}
        #! OccWorld Traverse 额外把 'scene-xxxx' 名称传给下游，用于场景级遍历和结果组织。
        metas.update(scene_name=scene_name)
        #! 同父类：此处变量名为 scene_token，但值实际是第 5 帧的 sample token。
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx))
        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        # import pdb; pdb.set_trace()
        return input_occs[:self.return_len], output_occs[self.offset:], metas

    def get_meta_info(self, scene_name, idx):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        #! 检测/运动 GT 与图像元信息使用同一个 OccWorld 固定参考帧（窗口末帧前 6 帧）。
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        '''gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
                print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        '''
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            #gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)

        return anns_results

    def get_image_info(self, scene_name, idx):
        #! 与父类相同：固定选择窗口末帧之前第 6 帧作为相机与位姿参考帧。
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t

            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            # cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            # focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))

        return input_dict
