from mmengine.registry import MODELS
from mmengine.model import BaseModule
import numpy as np
import torch.nn as nn, torch
import torch.nn.functional as F
from einops import rearrange
from copy import deepcopy
import torch.distributions as dist
from utils.metric_stp3 import PlanningMetric
import time

@MODELS.register_module()
class TransVQVAE(BaseModule):
    def __init__(self, vae, transformer, num_frames=10, offset=1,
                 pose_encoder=None, pose_decoder=None,
                 pose_actor=None, give_hiddens=False, delta_input=False, without_all=False):
        super().__init__()
        self.num_frames = num_frames
        self.offset = offset
        self.vae = MODELS.build(vae)
        self.transformer = MODELS.build(transformer)
        if pose_encoder is not None:
            self.pose_encoder = MODELS.build(pose_encoder)
        if pose_decoder is not None:
            self.pose_decoder = MODELS.build(pose_decoder)
        if pose_actor is not None:
            self.pose_actor = MODELS.build(pose_actor)
        self.give_hiddens = give_hiddens
        self.delta_input = delta_input
        self.planning_metric = None
        self.without_all = without_all
    def forward(self, x, metas=None):
        if hasattr(self, 'pose_encoder'): # stage2 train occworld
            if self.training:
                return self.forward_train_with_plan(x, metas)
            else:
                return self.forward_inference_with_plan(x, metas)
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)
    def forward_train(self, x):
        # given x: bs, f, h, w, d where f == num_frames + offset
        # output : ce_inputs: logits for the codebook
        # output : ce_labels: labels for the ce_inputs
        assert hasattr(self.vae, 'vqvae')
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        # z: bs*f, c, h, w

        # z: bs*f, h, w
        return output_dict


    def forward_inference(self, x):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        z_q_predict = z_q_predict.argmax(dim=1)
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)

        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)

        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou

        return output_dict


    def forward_train_with_plan(self, x, metas):
        assert hasattr(self.vae, 'vqvae')
        assert hasattr(self, 'pose_encoder')
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x) # (16 128 50 50) 训练完整occworld时，这部分无梯度
        z = self.vae.vqvae.quant_conv(z) # (16 128 50 50)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False) # (16 128 50 50)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs) # (1  16 50 50)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs) # (1 16 128 50 50)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]


        rel_poses, output_metas = self._get_pose_feature(metas, F-self.offset) # real_poses:(1 15 128) 自车位移(2)+自车指令(3) -> 编码为128维token

        z_q_predict, rel_poses = self.transformer(z_q[:, :self.num_frames], pose_tokens=rel_poses)

        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas


        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        # z: bs*f, c, h, w

        # z: bs*f, h, w
        return output_dict
    def forward_inference_with_plan(self, x, metas):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]


        rel_poses, output_metas = self._get_pose_feature(metas, F-self.offset)

        z_q_predict, rel_poses = self.transformer(z_q[:, :self.num_frames], pose_tokens=rel_poses)

        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas



        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        z_q_predict = z_q_predict.argmax(dim=1)
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)

        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)

        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou

        return output_dict

    def _get_pose_feature(self, metas=None, F=None):
        rel_poses, output_metas = None, None
        if hasattr(self, 'pose_encoder'):
            assert hasattr(self, 'pose_decoder')
            assert metas is not None
            output_metas = []
            for meta in metas:
                output_meta = dict()
                output_meta['rel_poses'] = meta['rel_poses'][self.offset:] # (15 2)
                output_meta['gt_mode'] = meta['gt_mode'][self.offset:] # (15 3) 指令
                output_metas.append(output_meta)


            rel_poses = np.array([meta['rel_poses'] for meta in metas]) # (1 16 2)
            gt_mode = np.array([meta['gt_mode'] for meta in metas]) # (1 16 3)




            gt_mode = torch.tensor(gt_mode).cuda()
            rel_poses = torch.tensor(rel_poses).cuda()# list of (num_frames+offsets, 2)
            if self.delta_input:
                rel_poses_pre = torch.cat([torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1]], dim=1)
                rel_poses = rel_poses - rel_poses_pre
            if F>self.num_frames:
                assert F == self.num_frames + self.offset
            else:
                assert F == self.num_frames
                gt_mode = gt_mode[:, :-self.offset, :] # (1 15 3)
                rel_poses = rel_poses[:, :-self.offset, :] # (1 16 2)

            rel_poses = torch.cat([rel_poses, gt_mode], dim=-1) # (1 15 5)
            #rel_poses = rearrange(rel_poses, 'b f d -> b f 1 d')
            rel_poses = self.pose_encoder(rel_poses.float()) # (1 15 128)
        return rel_poses, output_metas

    def forward_autoreg_with_pose(self, x, metas, start_frame=0, mid_frame=6,end_frame=12):
        #*==================== 1. 划分历史输入与未来监督区间 ====================#
        t0 = time.time()  # 记录整个编码与自回归过程的起始时间
        bs, F, H, W, D = x.shape  # x: (B,F,H,W,D)，连续多帧3D语义occupancy
        output_dict = {}  # 汇总预测、监督标签、元数据和耗时
        output_dict['input_occs'] = x[:, mid_frame-1:end_frame]  # 基线/可视化使用：最后1帧历史+未来GT
        output_dict['target_occs'] = x[:, mid_frame:end_frame]  # 待预测未来GT：(B,F_pred,H,W,D)

        #*==================== 2. 将3D Occupancy编码为离散场景Token ====================#
        #*【整体架构作用】这一节是“稠密3D场景”与“时序世界模型”之间的Tokenizer：
        #*   3D Occupancy -> VQ-VAE Tokenizer -> 2D离散场景Token -> Transformer预测未来Token。
        #* 它本身不预测未来，只负责把每帧庞大的200x200x16体素网格压缩成50x50个Token；
        #* 后面的Transformer才负责学习这些Token随时间的演化。完整OccWorld训练/评估时，
        #* Tokenizer通常加载第一阶段VQ-VAE权重并冻结，以保持Token语义空间固定。

        #* 三部分关系：Encoder提取连续特征 -> quant_conv投影到码本空间 -> Quantizer选最近码字。
        #* 1. 输入x为整数语义occupancy：(B,F,200,200,16)。先做类别嵌入、折叠高度维，
        #*    再用Encoder2D将H/W各压缩4倍；输出z是连续特征，不是离散Token。
        z, shape = self.vae.forward_encoder(x)  # z:(B*F,128,50,50)；shape记录200/100等解码尺寸
        #* 2. VQ比较距离要求每个位置的特征与码字具有相同维度e_dim。1x1卷积只在
        #*    每个位置混合/投影通道，不改变50x50空间尺寸；当前128->128，形状虽不变，
        #*    但特征被变换到适合与512个码字计算欧氏距离的可学习表示空间。
        z = self.vae.vqvae.quant_conv(z)  # (B*F,z_channels,50,50)->(B*F,e_dim,50,50)
        # *---------------------------------------------------------#
        #* 3. 对每帧50x50个位置，分别从512个128维码字中选欧氏距离最近者：
        #*    每个位置最终由一个0~511的整数Token表示，同时可从码本查回对应128维向量。
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = \
            self.vae.vqvae.forward_quantizer(z, is_voxel=False)  # is_voxel=False：量化2D BEV特征
        #*【两种Token表示及其用途】
        #* z_q: (B*F,128,50,50)，离散Token查码本所得的“向量形式”；历史部分作为Transformer输入，
        #*      自回归时预测Token也会查回这种向量并追加到上下文中，以继续预测下一帧。
        #* min_encoding_indices: (B*F,50,50)，离散Token的“整数编号形式”，取值0~511；
        #*      未来部分作为分类标签，监督Transformer在每个50x50位置预测正确码字。
        #* loss: VQ码本/承诺损失，主要用于第一阶段训练Tokenizer；完整世界模型阶段VAE通常被冻结。
        #* perplexity、min_encodings在当前实现中固定返回None，仅为兼容通用VQ接口。

        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)  # 恢复batch和时间维：(B,F,h,w)
        output_dict['ce_labels'] = min_encoding_indices[:, mid_frame:end_frame].detach().flatten(0,1)  # 未来真实token索引：(B*F_pred,h,w)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)  # 量化特征恢复为(B,F,C,h,w)
        z_q_predict = z_q[:, start_frame:mid_frame]  # 仅以历史真实token作为自回归初始上下文
        t1 = time.time()  # occupancy编码与量化结束时间

        #*==================== 3. 划分历史与未来自车运动元数据 ====================#
        #*【F和时间切片的含义】F是当前batch中每个样本包含的连续关键帧总数，不是特征维度。
        #* metas中的每个样本都按时间保存F步自车信息：
        #*   rel_poses: (F,2)，每一步相对运动(dx,dy)；gt_mode: (F,3)，每一步驾驶指令one-hot。
        #* 以评估配置start=0、mid=5、end=11为例：历史段[0,5)含5步，未来段[5,11)含6步。
        #*
        #*【为什么划分】历史自车运动和指令是世界模型的已知条件，与历史场景Token共同用于预测；
        #* 未来rel_poses是不能作为场景预测输入的GT，主要用于规划损失和评估预测轨迹。
        #* 未来gt_mode则被视为外部给定的高层导航指令：自回归时用它从右/左/直三条候选
        #* 位移中选择对应分支，但模型不会看到未来真实rel_poses。
        output_metas = []  # 未来F_pred=end-mid步的GT位移/指令，供规划监督和指标计算
        input_metas = []  # 历史F_hist=mid-start中的位移/指令，主要供输出记录及静态基线
        for meta in metas:  # 逐个batch样本截取历史自车元数据
            input_meta = dict()  # 当前样本的历史元数据容器
            input_meta['rel_poses'] = meta['rel_poses'][start_frame:mid_frame]  # (F_hist,2)历史逐步位移
            input_meta['gt_mode'] = meta['gt_mode'][start_frame:mid_frame]  # (F_hist,3)历史驾驶指令
            input_metas.append(input_meta)  # 加入batch历史元数据列表
        output_dict['input_metas'] = input_metas  # 保存历史元数据供后续输出/基线使用
        for meta in metas:  # 逐个batch样本截取待预测未来元数据
            output_meta = dict()  # 当前样本的未来元数据容器
            output_meta['rel_poses'] = meta['rel_poses'][mid_frame:end_frame]  # (F_pred,2)未来GT逐步位移
            output_meta['gt_mode'] = meta['gt_mode'][mid_frame:end_frame]  # (F_pred,3)未来已知驾驶指令
            output_metas.append(output_meta)  # 加入batch未来元数据列表
        output_dict['gt_poses_'] = np.array(
            [meta['rel_poses'] for meta in output_metas])  # 未来GT位移数组：(B,F_pred,2)
        rel_poses = np.array([meta['rel_poses'] for meta in metas])  # 堆叠batch：(B,F,2)
        gt_mode = np.array([meta['gt_mode'] for meta in metas])  # 堆叠batch：(B,F,3)
        gt_mode = torch.tensor(gt_mode).cuda()  # 驾驶模式转为GPU tensor

        #*==================== 4. 将历史自车位移与驾驶指令编码为Pose Token ====================#
        rel_poses = torch.tensor(rel_poses).cuda()  # 自车位移转为GPU tensor
        if self.delta_input:  # 可选：将输入位置序列再次转换成相邻差分；默认配置为False
            rel_poses_pre = torch.cat(torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1], dim=1)  # 原代码此处分支存在cat参数错误
            rel_poses = rel_poses - rel_poses_pre  # 当前位姿减前一位姿，得到增量输入
        rel_poses_sumed = rel_poses[:, start_frame:mid_frame]  # 保存历史位移；当前decode_pose基本未更新它
        rel_poses = torch.cat([rel_poses, gt_mode], dim=-1)  # 拼成(dx,dy)+3维驾驶模式，共5维
        rel_poses = rel_poses[:, start_frame:mid_frame]  # 只保留历史区间作为初始自车条件

        rel_poses = self.pose_encoder(rel_poses.float())  # * 原始自车输入token (B F 128) # 将5维自车状态编码为pose token
        rel_poses_state = rel_poses  # 保存不断追加预测结果的自车token序列
        z_q_list = []  # 收集每个未来时间步的场景token分类logits
        t2 = time.time()  # 元数据整理和自车状态编码结束时间
        poses_ = []  # 收集逐步选中的未来自车二维位移

        #*==================== 5. 联合自回归预测未来场景与自车运动 ====================#
        #*【时间边界】
        #* start_frame：初始历史窗口的起点；mid_frame：真实历史与未来预测的分界点；
        #* end_frame：预测区间的右边界（不包含）。所以模型以[start_frame, mid_frame)
        #* 的真实历史为起始条件，预测[mid_frame, end_frame)，总计end_frame-mid_frame帧。
        #* 例如start=0、mid=5、end=11：输入真实t0~t4，依次预测t5~t10，共6帧。
        #* 注意后面的未来结果切片直接使用[mid_frame:end_frame]，当前实现实际假定start_frame=0；
        #* OccWorld默认评估配置正是start_frame=0，若改成非0需同步修正相对下标。
        #*
        #*【自回归与回灌】每轮只采用当前序列的最后一项作为最新预测，再将它追加到输入：
        #*   第1轮：[真实t0~t4]                 -> 预测t5 -> 回灌t5；
        #*   第2轮：[真实t0~t4, 预测t5]         -> 预测t6 -> 回灌t6；
        #*   第3轮：[真实t0~t4, 预测t5, 预测t6] -> 预测t7 -> ……直到t10。
        #* 场景分支回灌预测的VQ码字向量；自车分支回灌预测(dx,dy)重新编码得到的Pose Token。
        for i in range(mid_frame, end_frame):  # i依次为5~10；每轮预测时间下标为i的那一帧
            # *==========================================================#
            #*【本轮输入】i既是当前上下文的结束位置，也是本轮待预测帧的时间下标：
            #* z_q_predict: (B,i-start_frame,128,50,50)，由历史真实场景码字向量和
            #*              前面各轮预测后回灌的场景码字向量组成；
            #* rel_poses_state: (B,i-start_frame,128)，与场景序列同步的自车Pose Token。
            #* 例如start=0、mid=5：首轮i=5输入5帧历史；次轮i=6输入5帧历史+1帧预测。
            z_q_, rel_poses_ = self.transformer.forward_autoreg_step(
                z_q_predict, pose_tokens=rel_poses_state,
                start_frame=start_frame, mid_frame=i)  # 联合预测下一帧场景token logits和自车token
            #*【本轮输出】函数对当前序列各位置产生错开一帧的预测结果：
            #* z_q_: (B,i-start_frame,512,50,50)，512为码本类别数，最后一项预测第i帧场景；
            #* rel_poses_: (B,i-start_frame,128)，最后一项是预测第i帧自车运动的隐藏特征。

            # *==========================================================#
            #* 后续只取[:, -1:]这项最新输出：场景分支选码字后回灌，Pose分支解码后回灌。
            z_q_list.append(z_q_[:, -1:])  # 保存第i帧场景logits，最后拼成全部未来帧监督输出
            #print(z_q_.shape)
            z_q_ = z_q_[:, -1:].clone().detach().argmax(dim=2)  # 仅取第i帧并选码字：(B,1,50,50)
            #print(z_q_.shape)
            z_q_ = self.vae.vqvae.get_codebook_entry(z_q_, shape=None)  # 码字编号->128维码字向量
            z_q_ = rearrange(z_q_, 'b f h w c-> b f c h w')  # (B,1,50,50,128)->(B,1,128,50,50)

            # *==========================================================#
            #*【场景回灌】把预测第i帧的码字向量追加到历史/预测上下文；下一轮将依赖它预测i+1。
            z_q_predict = torch.cat([z_q_predict, z_q_], dim=1)  # 时间长度从i-start变为i-start+1
            rel_poses = torch.cat([rel_poses, rel_poses_[:, -1:]], dim=1)  # 留存第i帧原始Pose隐藏特征
            rel_poses_state_, rel_poses_sumed, pose_ = self.decode_pose(
                rel_poses_[:, -1:], gt_mode[:,i:i+1], rel_poses_sumed)  # 解码3模态位移并按GT模式选分支
            poses_.append(pose_)  # 保存本时间步选中的预测位移(B,1,2)
            #*【自车回灌】decode_pose把选中的第i帧(dx,dy)+给定驾驶模式重新编码成128维Pose Token；
            #* 将它追加到自车上下文，使第i+1帧预测同时依赖此前预测的场景和自车运动。
            rel_poses_state = torch.cat(
                [rel_poses_state, rel_poses_state_], dim=1)  # 将预测位移重新编码并回灌下一步

        # *==========================================================#
        #* 循环结束时，两个上下文均为[真实历史, 全部未来预测]；下方只截取未来区间作为输出。
        poses_ = torch.cat(poses_, dim=1)  # t_mid~t_(end-1)逐步位移：(B,end-mid,2)
        output_dict['poses_'] = poses_  # 保存自回归过程中逐步选中的单模态轨迹
        t3 = time.time()  # 自回归循环结束时间
        z_q_predict = z_q_predict[:, mid_frame:end_frame]  # 去掉真实历史，仅保留预测t_mid~t_(end-1)
        rel_poses = rel_poses[:, mid_frame:end_frame]  # 同样仅保留未来Pose隐藏特征供最终3模态解码

        #*==================== 6. 解码未来三模态自车轨迹 ====================#
        #assert False, f'z_q_predict.shape: {z_q_predict.shape}, rel_poses.shape: {rel_poses.shape}, {output_dict["target_occs"].shape}'
        # print(z_q_predict.shape, rel_poses.shape)
        pose_decoded = self.pose_decoder(rel_poses)  # 解码未来每一步的3模态(dx,dy)：(B,F_pred,3,2)
        output_dict['pose_decoded'] = pose_decoded  # 供规划损失和ST-P3指标选择对应模式
        output_dict['output_metas'] = output_metas  # 保存同一未来区间的GT位移和驾驶模式

        #*==================== 7. 解码未来场景Token并生成3D Occupancy ====================#
        z_q = torch.cat(z_q_list, dim=1)  # 拼接未来各步的场景token分类logits
        #print(z_q.shape)
        output_dict['ce_inputs'] = z_q.flatten(0, 1)  # (B*F_pred,N_code,h,w)，用于token交叉熵
        z_q_predict = z_q_predict.flatten(0, 1)  # (B*F_pred,C,h,w)，准备逐帧解码occupancy
        # output_dict['ce_inputs'] = z_q_predict
        # z_q_predict = z_q_predict.argmax(dim=1)
        # z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        # z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)  # 码字维度投影回VAE解码器通道

        z_q_predict = self.vae.forward_decoder(
            z_q_predict, shape, output_dict['target_occs'].shape)  # 解码为(B,F_pred,H,W,D,18)
        output_dict['logits'] = z_q_predict  # 保存每个体素的18类语义logits
        pred = z_q_predict.argmax(dim=-1).detach().cuda()  # 取最大类别得到语义occupancy预测
        output_dict['sem_pred'] = pred  # (B,F_pred,H,W,D)，用于多步语义mIoU
        pred_iou = deepcopy(pred)  # 复制一份并转换成occupied/free二值结果

        pred_iou[pred_iou!=17] = 1  # 类别17以外均视为被占用
        pred_iou[pred_iou==17] = 0  # 类别17是free，转换为未占用0
        output_dict['iou_pred'] = pred_iou  # 二值occupancy预测，用于occupied IoU


        #*==================== 8. 可选静态基线与推理耗时统计 ====================#
        if self.without_all:  # 消融/基线模式：不使用世界模型预测，直接复制历史最后一帧
            #output_dict['pose_decoded'] =
            output_dict['sem_pred'] = output_dict['input_occs'][:, 0:1].repeat(
                1, end_frame-mid_frame, 1, 1, 1)  # 复制最后历史occupancy作为所有未来预测
            pred_iou = deepcopy(output_dict['sem_pred'])  # 复制语义结果用于二值化
            pred_iou[pred_iou!=17] = 1  # 非free类别置为occupied
            pred_iou[pred_iou==17] = 0  # free类别置为unoccupied
            output_dict['iou_pred'] = pred_iou  # 覆盖世界模型的二值预测
            output_dict['pose_decoded'] = torch.tensor(
                [meta['rel_poses'] for meta in input_metas])[:,-1:].unsqueeze(2).repeat(
                    1, end_frame-mid_frame, 3, 1)  # 复制最后历史位移为3模式未来轨迹
        output_dict['time'] = {
            'encode':t1-t0,  # occupancy编码与量化耗时
            'mid':t2-t1,  # 元数据整理与pose编码耗时
            'autoreg':t3-t2,  # 全部未来帧自回归耗时
            'total':t3-t0,  # 编码、准备和自回归总耗时（不含最终occupancy解码）
            'per_frame':t1-t0+(t3-t2)/(end_frame-mid_frame)}  # 原实现估算的单帧时间
        return output_dict  # 返回预测、GT token、未来元数据及耗时统计



    def decode_pose(self, pose, gt_mode, rel_poses_sumed):
        pose = self.pose_decoder(pose)
        # pose:b, f, 3, 2
        # mode:b, f, 3
        # b, f, 2
        bs, num_frames, num_modes, _ = pose.shape
        #gt_mode_ = gt_mode.unsqueeze(-1).repeat(1, 1, 1, 2)
        pose = pose[gt_mode.bool()].reshape(bs, num_frames, 2)
        pose_decoded = pose.clone().detach()
        '''if not self.delta_input:
            pose = pose+rel_poses_sumed[:, -1:]
            rel_poses_sumed = torch.cat([rel_poses_sumed, pose], dim=1)'''
        pose = torch.cat([pose, gt_mode], dim=-1)
        pose = self.pose_encoder(pose.float())
        return pose, rel_poses_sumed, pose_decoded
    def forward_autoreg(self, x, metas=None, start_frame=0, mid_frame=6,end_frame=12):

        pass
    def generate_inference(self, x):
        #import pdb; pdb.set_trace()
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        z_q_predict = z_q_predict.permute(0, 2, 3, 1)
        cata_distribution = dist.Categorical(logits=(z_q_predict-z_q_predict.min())/(z_q_predict.max()-z_q_predict.min()))
        import pdb;pdb.set_trace()
        z_q_predict = cata_distribution.sample()
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)

        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)

        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou

        return output_dict

    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3"""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            'plan_L2_1s_single':0,
            'plan_L2_2s_single':0,
            'plan_L2_3s_single':0,
            'plan_obj_col_1s_single':0,
            'plan_obj_col_2s_single':0,
            'plan_obj_col_3s_single':0,
            'plan_obj_box_col_1s_single':0,
            'plan_obj_box_col_2s_single':0,
            'plan_obj_box_col_3s_single':0,

        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                traj_L2_single = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, cur_time-1:cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, cur_time-1:cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                obj_coll_single, obj_box_coll_single = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, cur_time-1:cur_time].detach(),
                    gt_ego_fut_trajs[:, cur_time-1:cur_time],
                    occupancy[:, cur_time-1:cur_time])
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_L2_{}s_single'.format(i+1)] = traj_L2_single
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
                metric_dict['plan_obj_col_{}s_single'.format(i+1)] = obj_coll_single.item()
                metric_dict['plan_obj_box_col_{}s_single'.format(i+1)] = obj_box_coll_single.item()


            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_L2_{}s_single'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0

        return metric_dict

    def autoreg_for_stp3_metric(self, x, metas,
                                start_frame=0, mid_frame=6,end_frame=12):
        # [start_frame, mid_frame) 为历史输入，[mid_frame, end_frame) 为自回归预测区间。
        # “自回归”是指：模型每预测出一个未来帧，就把这个预测结果作为下一次预测的输入，继续预测更远的未来。
        output_dict = self.forward_autoreg_with_pose(
            x, metas, start_frame, mid_frame, end_frame)  # 同时预测未来 occupancy 和自车位移
        pred_ego_fut_trajs = output_dict['pose_decoded']  # (B, F_pred, 3, 2)：3 种模式的 (dx,dy)
        gt_mode = torch.tensor([meta['gt_mode'] for meta in output_dict['output_metas']])  # (B,F_pred,3)，GT 指令 one-hot
        bs, num_frames, num_modes, _ = pred_ego_fut_trajs.shape  # B、未来帧数、模式数(默认3)、xy维度
        pred_ego_fut_trajs = pred_ego_fut_trajs[gt_mode.bool()].reshape(bs, num_frames, 2)  # 按 GT 指令选中每步对应分支：(B,F_pred,3,2)->(B,F_pred,2)
        pred_ego_fut_trajs = torch.cumsum(pred_ego_fut_trajs, dim=1).cpu()  # 逐步位移累加为相对预测起点的未来位置
        gt_ego_fut_trajs = torch.tensor([meta['rel_poses'] for meta in output_dict['output_metas']])  # GT 逐步位移 (B,F_pred,2)
        gt_ego_fut_trajs = torch.cumsum(gt_ego_fut_trajs, dim=1).cpu()  # GT 也累加为未来位置，与预测采用相同表示
        assert len(metas) == 1, f'len(metas): {len(metas)}'  # ST-P3 规划指标目前只支持 B=1
        gt_bbox = metas[0]['gt_bboxes_3d']  # 参考帧周围目标的 3D GT box，用于碰撞评估
        gt_attr_labels = torch.tensor(metas[0]['attr_labels'])  # 目标未来轨迹/mask/goal/状态/yaw 的拼接标签
        fut_valid_flag = torch.tensor(metas[0]['fut_valid_flag'])  # 未来监督是否完整；原实现读取后未传给指标函数
        # import pdb;pdb.set_trace()
        metric_stp3 = self.compute_planner_metric_stp3(
            pred_ego_fut_trajs, gt_ego_fut_trajs,
            gt_bbox, gt_attr_labels[None], True)  # 计算1/2/3秒 L2及碰撞率；原代码固定按有效样本处理
        output_dict['metric_stp3'] = metric_stp3  # 交给 eval_metric_stp3.py 跨样本累加并求平均
        return output_dict  # 返回 occupancy/轨迹预测、中间监督、耗时和规划指标
