import torch
from torch.nn import functional as F
from torch import nn
from .modules import FFN


from mmengine.registry import MODELS
from mmengine.model import BaseModule
from einops import rearrange


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
class IdentityUnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 1, 0)
        #self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.ln(input)
        output = self.conv1(output)
        output = self.act(output)
        return output

@MODELS.register_module()
class PlanUAutoRegTransformer(BaseModule):
    """OccWorld的空间-时间生成Transformer。

    输入包含两条同步序列：
      1. scene tokens: (B,F,C,H,W)，每帧H*W个VQ场景码字向量；
      2. pose tokens:  (B,F,C)，每帧一个自车运动/驾驶指令向量。
    模型在多个BEV尺度上交替执行固定空间位置的时间注意力、场景空间卷积聚合，
    以及自车token对场景token的空间注意力，最后输出下一时刻的码字分类logits
    和自车pose特征。外层TransVQVAE负责逐帧回灌输出，实现自回归预测。
    """
    def __init__(
        self,
        num_tokens,
        num_frames,
        num_layers,
        img_shape,
        pose_shape,
        tpe_dim=10,
        output_channel=1024,
        channels=[1,2,3],
        ffn_dims=None,
        temporal_attn_layers=1,
        pose_attn_layers=1,
        num_heads=8,
        pose_output_channel=None,
        conditional=True,
        tokens_untouched=False,
        add_aggregate=False,
        learnable_queries=True,
        without_multiscale=False,
        without_spatial_attn=False,
        without_pose_spatial_attn=False,
        without_pose_temporal_attn=False,
        without_temporal_attn=False) -> None:
        super().__init__()
        #*==================== 1. 保存基础配置与定义预测Query ====================#
        if without_multiscale:
            assert len(channels) == 2
        self.num_tokens = num_tokens  # 每个时间位置参与因果mask的token组数；默认1
        self.num_frames = num_frames  # 模型允许处理的最大连续帧数
        self.num_layers = num_layers  # 保留的配置字段；实际层数由各attention参数控制
        self.channels = channels  # U-Net各尺度通道数，如(128,256,512)
        self.learnable_queries = learnable_queries  # 是否用独立可学习query预测，而非直接复制输入token
        if self.learnable_queries:
            #! self.queries不是未来场景Token，而是每个BEV位置用于承载“下一时刻预测”的可学习初始向量。
            #! 它会在forward中复制到每个时间位置，再通过时间编码指定各位置实际要预测的时刻。
            self.queries = nn.Embedding(img_shape[1]*img_shape[2], img_shape[0])
        self.offset = 1 if conditional else 0  # conditional=True时执行“给定t0~t(F-1)，逐步预测下一帧”的条件生成
        #* 场景时间Embedding表共有num_frames+offset项：输入token最多使用索引[0,num_frames-1]；
        #* offset=1时额外预留索引num_frames，供最后一个query表示“下一时刻”使用。多出的1只是
        #* 一个可学习时间向量，不是额外输入帧；配置要求tpe_dim与场景token通道C一致，才能直接相加。
        self.temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        if self.learnable_queries:
            #! self.pose_queries同样是“下一时刻自车运动”的预测槽初始向量，不是预先填入的未来位姿。
            #! 每帧只有一个自车Query；pose_shape通常为(1,128)。
            self.pose_queries = nn.Embedding(pose_shape[0], pose_shape[1])
        #* 自车分支采用同样的时间错位规则，但独立学习Embedding参数，避免与场景分支强制共享。
        self.pose_temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        self.pose_attn_layers = pose_attn_layers if pose_attn_layers is not None else temporal_attn_layers

        #*==================== 2. 建立多尺度U-Net编码/解码容器 ====================#
        # scene分支：en/de分别保存下采样侧和上采样侧的时间注意力与空间聚合模块。
        self.temporal_attentions_en = nn.ModuleList([])
        self.temporal_attentions_de = nn.ModuleList([])
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # pose分支：每个尺度同时对历史pose做时间注意力，并让pose query关注当前场景。
        self.pose_attn_en = nn.ModuleList([])
        self.pose_en = nn.ModuleList()
        #self.pose_temporal_attn_en = nn.ModuleList([])
        self.pose_attn_de = nn.ModuleList([])
        self.pose_de = nn.ModuleList()
        self.pose_up = nn.ModuleList()
        #self.pose_temporal_attn_de = nn.ModuleList([])

        # stride=2卷积实现论文中的2x2窗口聚合；反卷积负责恢复空间分辨率。
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        up_down_sample_params = dict(kernel_size=2, stride=2, padding=0)  # H/W减半，token数约变为1/4
        self.up_down_sample_params = up_down_sample_params
        self.unfold_params = dict(kernel_size=[1, 2], stride=[1, 2], padding=[0, 0])

        C, H, W = img_shape  # 默认场景token特征图为(128,50,50)
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers-1):
            cH = (cH) // 2
            cW = (cW) // 2
            Hs.append(cH)
            Ws.append(cW)
        # 默认channels=(128,256,512)时，多尺度空间尺寸约为50x50、25x25、12x12。
        pre_c = C
        #*==================== 3. 构建U-Net下采样侧的场景与Pose模块 ====================#
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                    ]))
                else:
                    # 每个BEV位置沿F帧单独做MultiheadAttention，再接FFN与残差归一化。
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.temporal_attentions_en.append(temporal_attn_layer)
            if without_spatial_attn:
                self.encoders.append(nn.Sequential(IdentityUnetBlock((cH, cW), pre_c, channel),
                                                   IdentityUnetBlock((cH, cW), channel, channel)))
            else:
                # UnetBlock用3x3二维卷积聚合同一帧内相邻场景token的空间信息。
                self.encoders.append(nn.Sequential(UnetBlock((cH, cW), pre_c, channel, True),
                                               UnetBlock((cH, cW), channel, channel, True)))
            if without_multiscale:
                self.downsamples.append(Identity())
            else:
                # 2x2、stride=2：H/W各减半，实现论文中的多尺度world token。
                self.downsamples.append(nn.Conv2d(channel, channel, **up_down_sample_params))
            self.pose_en.append(nn.Sequential(
                nn.Linear(pre_c, channel),nn.ReLU(),
                nn.Linear(channel, channel),nn.ReLU(),
            ))
            pose_attn_layer = nn.ModuleList()
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                ]))
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                ]))
                else:
                    # pose模块依次包含：跨时间注意力、pose对场景的空间注意力、FFN。
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.pose_attn_en.append(pose_attn_layer)
            pre_c = channel
        channel = channels[-1]
        #*==================== 4. 构建最低分辨率瓶颈层 ====================#
        if without_multiscale:
            if without_spatial_attn:
                self.mid = nn.Sequential(
                    IdentityUnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                    IdentityUnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
                )
            else:
                self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                UnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
            )
        else:
            self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[-1], Ws[-1]), pre_c, channel, True),
                UnetBlock((channel, Hs[-1], Ws[-1]), channel, channel, True),
            )
        self.pose_mid = nn.Sequential(
            nn.Linear(pre_c, channel),nn.ReLU(),
            nn.Linear(channel, channel),nn.ReLU(),
        )
        pre_c = channel
        #*==================== 5. 构建U-Net上采样侧与跳跃融合模块 ====================#
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            channel_agg = channel if add_aggregate else channel * 2  # 默认拼接skip特征，故通道翻倍
            if without_multiscale:
                self.upsamples.append(Identity())
            else:
                # 反卷积逐级恢复BEV分辨率，如12->24(再padding到25)->50。
                self.upsamples.append(nn.ConvTranspose2d(pre_c, channel, **up_down_sample_params))
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.temporal_attentions_de.append(temporal_attn_layer)
            if without_spatial_attn:
                self.decoders.append(nn.Sequential(
                    IdentityUnetBlock((channel_agg, cH,cW), channel_agg, channel, True),
                    IdentityUnetBlock((channel, cH,cW), channel, channel, True)
                ))
            else:
                self.decoders.append(
                    nn.Sequential(
                        UnetBlock((channel_agg, cH,cW),
                                channel_agg, channel, True),
                        UnetBlock((channel, cH,cW),
                                channel,
                                channel,
                                True)
                    )
                )
            pose_attn_layer = nn.ModuleList()
            self.pose_up.append(nn.Linear(pre_c, channel))
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                ]))
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                ]))
                else:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.pose_attn_de.append(pose_attn_layer)
            self.pose_de.append(nn.Sequential(
                nn.Linear(channel_agg, channel),nn.ReLU(),
                nn.Linear(channel, channel),nn.ReLU(),
            ))
            pre_c = channel

        #*==================== 6. 定义场景Token与自车Token输出头 ====================#
        # output_channel默认512：每个50x50位置输出对512个VQ码字的分类logits。
        self.conv_out = nn.Conv2d(pre_c, output_channel, 3, 1, 1)

        # pose_out输出供PoseDecoder使用的自车隐藏特征，默认维度128。
        self.pose_out = nn.Linear(pre_c, pose_output_channel if pose_output_channel is not None else output_channel)

        self.tokens_untouched = tokens_untouched

        #*==================== 7. 构建GPT式因果时间注意力Mask ====================#
        if tokens_untouched:
            assert all([ch == channels[0] for ch in channels])
            for scale in range(len(channels) - 1):
                num_tokens = self.unfold_params['kernel_size'][scale] ** 2
                attn_mask = torch.zeros(num_frames, num_frames * num_tokens, dtype=torch.bool)
                for i_frame in range(num_frames):
                    start = i_frame * num_tokens + num_tokens if conditional else i_frame * num_tokens
                    attn_mask[i_frame, start:] = True
                self.register_buffer(f'attn_mask_{scale}', attn_mask, False)
        else:
            # 默认num_tokens=1时mask形状为(F,F)：第t个query只能读取不晚于t的输入，
            # conditional=True又令query时间位置整体后移一格，从而学习历史->下一帧预测。
            attn_mask = torch.zeros(num_frames * num_tokens, num_frames * num_tokens, dtype=torch.bool)
            for i_frame in range(num_frames):
                start1 = i_frame * num_tokens
                start2 = start1 + num_tokens if conditional else start1
                attn_mask[start1: (start1 + num_tokens), start2:] = True
            self.register_buffer('attn_mask', attn_mask, False)

    def forward(self, tokens, pose_tokens):
        #*==================== 并行训练前向：已知整段历史，预测各时刻下一帧 ====================#
        # 训练时一次处理固定F帧；query使用t+1时间embedding，token使用t时间embedding，
        # 再配合因果mask并行学习z_t -> z_(t+1)。推理则调用下方forward_autoreg_step。
        #import pdb;pdb.set_trace()
        # tokens: bs, f, c, h, w
        # pose_tokens, bs, f, c
        bs, F, C, H, W = tokens.shape
        assert F == self.num_frames
        tokens = rearrange(tokens, 'b f c h w -> b f h w c')
        if self.learnable_queries: # * False
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            queries = tokens # * 历史+当前+[预测] 场景token

        #* token位置k表示已知tk，使用时间编码k；query位置k负责预测t(k+1)，使用时间编码k+1。
        #* 因此最后一个query可汇聚此前所有有效token并预测下一帧，而不是把输入整体平移F帧。
        queries = queries + self.temporal_embeddings.weight[None, self.offset:, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :self.num_frames, None, None, :].expand(
            bs, -1, H, W, -1)

        if self.learnable_queries:
            pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
        else:
            pose_queries = pose_tokens
        pose_queries = pose_queries + self.pose_temporal_embeddings.weight[None, self.offset:, :].expand(
            bs, -1, -1)
        pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[None, :self.num_frames, :].expand(
            bs, -1, -1)


        encoder_outs_tokens = []
        encoder_outs_queries = []
        encoder_outs_pose_tokens = []
        encoder_outs_pose_queries = []

        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(self.temporal_attentions_en, self.encoders, self.downsamples, self.pose_attn_en, self.pose_en):
            b, f, h, w, c = tokens.shape

            for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_en:
                pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                pose_queries = pose_temporal_norm(pose_queries)
                #b, f, h, w, c = queries.shape
                pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                pose_queries = spatial_norm(pose_queries)

                pose_queries = pose_queries + ffn(pose_queries)
                pose_queries = ffn_norm(pose_queries)
                pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                queries = rearrange(queries, '(b f) (h w) c -> b f h w c', b=b, f=f, h=h, w=w)

            pose_queries = pose_en(pose_queries)
            pose_tokens = pose_en(pose_tokens)
            encoder_outs_pose_queries.append(pose_queries)
            encoder_outs_pose_tokens.append(pose_tokens)

            queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            #queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = cross_norm(queries)

                queries = queries + ffn(queries)
                queries = ffn_norm(queries)

            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f)
        b, f, h, w, c = queries.shape
        queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)

        pose_queries = self.pose_mid(pose_queries)
        pose_tokens = self.pose_mid(pose_tokens)

        # queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
        # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f)
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens, pose_attn_de, pose_de_, encoder_out_pose_queries, encoder_out_pose_tokens, pose_up in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1], self.pose_attn_de, self.pose_de,
                                                                        encoder_outs_pose_queries[::-1], encoder_outs_pose_tokens[::-1], self.pose_up):
            queries = up(queries)
            tokens = up(tokens)

            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = cross_norm(queries)

                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)


            pose_queries = pose_up(pose_queries)
            pose_tokens = pose_up(pose_tokens)
            pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)
            pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)

            for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_de:
                pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                pose_queries = pose_temporal_norm(pose_queries)
                #b, f, h, w, c = queries.shape
                pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                #queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                queries = rearrange(queries, '(b f) c h w -> (b f) (h w) c', b=b, f=f, h=h, w=w)
                pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                pose_queries = spatial_norm(pose_queries)

                pose_queries = pose_queries + ffn(pose_queries)
                pose_queries = ffn_norm(pose_queries)
                queries = rearrange(queries, '(b f) (h w) c -> (b f) c h w', b=b, f=f, h=h, w=w)
                pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
            pose_queries = pose_de_(pose_queries)
            pose_tokens = pose_de_(pose_tokens)
            queries = decoder(queries)
            tokens = decoder(tokens)

        queries = self.conv_out(queries)
        pose_queries = self.pose_out(pose_queries)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)

        return queries,  pose_queries

    def forward_autoreg(self, tokens, pose_tokens, start_frame=0, mid_frame=6, end_frame=12):
        #* 完整自回归包装：反复调用单步预测，并把最新scene/pose结果追加回历史序列。
        tokens = tokens[:, start_frame:mid_frame]
        pose_tokens = pose_tokens[:, start_frame:mid_frame]
        for i in range(mid_frame, end_frame):
            token, pose_token = self.forward_autoreg_step(tokens, pose_tokens, start_frame, i)

            tokens = torch.cat([tokens, token[:, -1:]], dim=1)
            pose_tokens = torch.cat([pose_tokens, pose_token[:, -1:]], dim=1)
        b, f, c, h, w = tokens.shape
        queries = rearrange(tokens, 'b f c h w -> (b f) c h w')
        queries = self.conv_out(queries)
        pose_queries = self.pose_out(pose_tokens)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)

        return queries[:,mid_frame:end_frame], pose_queries[:,mid_frame:end_frame]

    def forward_autoreg_step(self, tokens, pose_tokens, start_frame=0, mid_frame=6):
        """根据当前可见历史，联合预测场景Token和自车Pose Token。

        与TransVQVAE.forward_autoreg_with_pose()调用处对应：
            tokens:      (B,F_ctx,C,H,W)，历史真实场景码字向量与此前预测回灌结果；
                         默认C=128、H=W=50。
            pose_tokens: (B,F_ctx,C)，与场景帧同步的历史/回灌自车Pose Token；
                         默认C=128。
            start_frame: 当前滑动上下文的起始下标，默认0。
            mid_frame:   当前上下文的结束下标（左闭右开），同时也是本轮待预测帧下标。

        首轮示例（start=0、mid=5）：
            tokens.shape      = (B,5,128,50,50)
            pose_tokens.shape = (B,5,128)
        下一轮已回灌一次预测，mid=6，两个输入的时间长度随之变为6。

        返回：
            queries:      (B,F_used,512,50,50)，各位置的下一时刻VQ码字分类logits；
            pose_queries: (B,F_used,128)，各位置的下一时刻自车隐藏特征。
        虽然函数名为step，它会返回当前F_used个位置的错位预测；外层仅取[:, -1:]
        作为本轮最新的第mid_frame帧预测，再将其回灌以生成下一帧。
        """
        #!==================== Query/Token概念：避免把Query误认为预填充的未来帧 ====================#
        #! tokens：当前已经可用的场景上下文。首轮为真实历史/当前帧；后续还包括此前预测并回灌的未来帧。
        #! pose_tokens：与tokens逐帧对齐的已有自车运动上下文，后续轮次同样包含预测回灌结果。
        #! queries：承载“下一时刻场景预测”的中间特征槽，不是真实未来Token，也没有提前填入未来GT。
        #! pose_queries：承载“下一时刻自车运动预测”的中间特征槽，同样不是预填充的未来位姿。
        #! 当learnable_queries=False时，Query虽然复制Token进行初始化，但加入错位时间编码并经过网络后，
        #! 其语义变为下一时刻的预测；例如[t0,t1,t2,t3,t4]初始化出的5个Query分别预测[t1,...,t5]。
        #! 函数会并行计算这些下一时刻预测，但外层自回归只取最后一项t5并回灌，再调用本函数预测t6。

        #*==================== 1. 截取当前可见的历史Scene/Pose Token ====================#
        #! 这里截取的是“已有上下文”，而不是包含空白t5的定长序列；待预测t5由最后一个Query负责承载。
        # tokens:(B,F_all,C,H,W)，pose_tokens:(B,F_all,C)；本函数只向上下文末尾推进一帧。
        bs, F, C, H, W = tokens.shape
        #assert F == self.num_frames
        tokens = tokens[:, start_frame:mid_frame]  # 保留[start_frame,mid_frame)内当前可用场景历史
        pose_tokens = pose_tokens[:, start_frame:mid_frame]  # 截取完全同步的自车历史
        bs, F, C, H, W = tokens.shape  # F更新为本次单步预测实际使用的历史长度
        tokens = rearrange(tokens, 'b f c h w -> b f h w c')  # 通道后置，便于时间attention

        #*==================== 2. 构造下一时刻Query并加入错位时间Embedding ====================#
        #* 2.1 构造与F帧输入逐位置对应的场景Query。
        if self.learnable_queries:
            #! 可学习Query仅提供BEV空间位置相关的预测槽初值，不含任何真实或预测的未来场景内容。
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            #! 复制已有Token只是为下一时刻预测提供较好的初值，并不表示输出仍是该已有Token。
            queries = tokens  # 默认：以[t0,...,t(F-1)]初始化分别负责预测[t1,...,tF]的Query

        #* 2.2 为已知Token和预测Query加入相差一个时间步的时间编码。
        #* 当前调用共有F帧上下文：tokens位置0~F-1表示当前已知的t0~t(F-1)，使用时间编码[0,F-1]；
        #* queries位置0~F-1分别负责预测t1~tF，因此使用时间编码[1,F]。这里“错开一帧”表示：
        #* 每个query预测对应token的下一时刻，并非只使用前一帧；在因果注意力下，最后一个query仍能
        #* 汇聚t0~t(F-1)全部上下文来预测tF。历史长度F和单轮预测步长1是两个不同概念。
        #! 例：输入Token=[真实t0,...,真实t4]时，Query目标=[预测t1,...,预测t5]；前四项也是
        #! 网络重新计算的预测，而非直接变成真实t1~t4。外层仅取最后一项预测t5并回灌，下一轮
        #! Token上下文才变成[真实t0~t4,预测t5]，然后以相同方式产生预测t6。
        queries = queries + self.temporal_embeddings.weight[None, self.offset:F+self.offset, None, None, :].expand(bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :F, None, None, :].expand(bs, -1, H, W, -1)
        if self.learnable_queries:
            pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
        else:
            #! 与场景分支相同：复制已有Pose Token只是初始化下一时刻Pose Query，不是填入未来位姿。
            pose_queries = pose_tokens  # 默认以已有自车运动初始化分别负责下一时刻预测的Pose Query
        #! 自车分支也执行一步错位：Pose Token表示已知t0~t(F-1)，Pose Query负责预测t1~tF。
        pose_queries = pose_queries + self.pose_temporal_embeddings.weight[None, self.offset:F+self.offset, :].expand(bs, -1, -1)
        pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[None, :F, :].expand(bs, -1, -1)


        #*==================== 3. 初始化U-Net各尺度跳跃连接缓存 ====================#
        encoder_outs_tokens = []
        encoder_outs_queries = []
        encoder_outs_pose_tokens = []
        encoder_outs_pose_queries = []

        #*==================== 4. 多尺度编码：时序建模、空间交互与2倍下采样 ====================#
        #* zip后每轮对应一个U-Net尺度；默认依次处理50x50/128通道和25x25/256通道，
        #* 再把结果下采样到约12x12/512通道的瓶颈。scene与pose两条分支始终同步变换。
        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(self.temporal_attentions_en, self.encoders, self.downsamples, self.pose_attn_en, self.pose_en):
            b, f, h, w, c = tokens.shape  # 当前尺度场景序列：(B,F,H,W,C)

            #* 4.1 自车分支：先沿时间聚合历史pose，再让每帧pose query关注该帧全部场景位置。
            for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_en:
                #!【自车时间注意力：Pose Query读取历史Pose Token】
                # * Q=pose_queries：(B,F,C)，第k个query表示要预测时刻t(k+1)的自车运动；
                # * K/V=pose_tokens：(B,F,C)，第k个token保存已知时刻tk的自车运动信息。
                # * 信息交互范围：同一个样本内，不同时间的自车token之间交互，不涉及BEV场景token；
                # * 在因果mask约束下，第k个pose query只能读取其允许范围内的当前/更早pose token，
                # * 从而综合历史运动趋势预测下一时刻，不能读取相对该query而言的未来自车信息。
                #! 更新对象：仅更新Q端pose_queries；K/V端pose_tokens只被读取，不会被这次Attention改写。
                pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]  # 注意力结果残差加回query
                pose_queries = pose_temporal_norm(pose_queries)  # 时间注意力后的LayerNorm
                #b, f, h, w, c = queries.shape
                pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')  # (B,F,C)->(B*F,1,C)，每帧1个自车query
                queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')  # (B,F,H,W,C)->(B*F,H*W,C)
                #!【自车-场景空间注意力：Pose Query读取同帧全部场景Query】
                # * Q=pose_queries：(B*F,1,C)，每个样本、每一帧只有1个自车query；
                # * K/V=queries：(B*F,H*W,C)，包含该帧H*W个BEV位置的场景query。
                # * 信息交互范围：自车query同时读取同一帧全部BEV位置，聚合道路、车辆、障碍物等环境信息；
                # * 不同帧已折叠进B*F批维，彼此不会在本注意力中交互，所以无需使用时间因果mask。
                #! 更新对象：仅更新pose_queries，使自车表示融合场景信息；场景queries只被读取。
                #! 因而这里是单向的“场景->自车”，并没有使用自车信息反向更新场景Query。
                pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]  # 无时间mask，因为交互发生在同一帧内
                pose_queries = spatial_norm(pose_queries)  # 场景空间注意力后的LayerNorm

                # *【自车FFN】每个pose query独立进行通道变换，不与其他时间或空间token交换信息；
                # * 更新对象仍然只有pose_queries，并通过残差连接保留注意力聚合得到的信息。
                pose_queries = pose_queries + ffn(pose_queries)  # FFN非线性变换，并使用残差连接
                pose_queries = ffn_norm(pose_queries)  # FFN输出归一化
                pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)  # 恢复自车序列(B,F,C)
                queries = rearrange(queries, '(b f) (h w) c -> b f h w c', b=b, f=f, h=h, w=w)  # 恢复场景网格(B,F,H,W,C)

            pose_queries = pose_en(pose_queries)  # MLP把预测pose query投影到当前尺度通道数
            pose_tokens = pose_en(pose_tokens)  # 已知pose token同步投影，保证Q与K/V维度一致
            encoder_outs_pose_queries.append(pose_queries)  # 保存当前尺度query，供U-Net解码端skip融合
            encoder_outs_pose_tokens.append(pose_tokens)  # 保存当前尺度token，供U-Net解码端skip融合

            #* 4.2 场景分支：把每个固定BEV位置跨F帧组成一个时间序列，执行因果时间注意力。
            queries = rearrange(queries, 'b f h w c -> (b h w) f c')  # 每个BEV位置形成一条待预测时间序列
            tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')  # 同一BEV位置的已知历史序列作为K/V
            #queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                #!【场景时间注意力：固定BEV位置的场景Query读取该位置的历史场景Token】
                # * Q=queries：(B*H*W,F,C)，第k个query表示该BEV位置在t(k+1)时刻的待预测状态；
                # * K/V=tokens：(B*H*W,F,C)，保存同一BEV位置在t0~t(F-1)的已有场景状态。
                # * 信息交互范围：每个BEV位置分别沿时间轴交互；位置(x,y)只读取自身各历史时刻，
                # * 不会在本注意力中直接读取其他空间位置，跨位置交互留给后面的2D UnetBlock完成。
                # * 因果mask保证第k个场景query只读取允许的当前/更早场景token，不读取未来状态。
                #! 更新对象：仅更新Q端queries以形成下一时刻预测特征；K/V端tokens只被读取。
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]  # 因果时间注意力+残差
                queries = cross_norm(queries)  # 时间注意力输出LayerNorm

                # *【场景FFN】每个“空间位置-时间”query独立做通道变换，不产生额外token间交互；
                # * 更新对象仍为queries，残差连接用于保留时间注意力得到的演化信息。
                queries = queries + ffn(queries)  # 对各位置各时刻独立执行通道FFN并残差相加
                queries = ffn_norm(queries)  # FFN输出LayerNorm

            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)  # (B*H*W,F,C)->(B*F,C,H,W)，准备2D空间卷积
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)  # 已知token也转成同样的2D CNN格式
            #* 4.3 使用U-Net卷积补充场景空间交互，并建立多尺度空间表示。
            #* 前面的场景时间Attention只让同一BEV位置(x,y)在不同时间之间交互，不会让相邻空间
            #* 位置交换信息；自车-场景Attention也只更新Pose Query。因此仍需2D卷积建模空间关系。
            # 例如车辆从相邻网格驶入当前网格、道路结构连续性和目标占据范围，都需要结合邻域信息。
            # 两个3x3 UnetBlock逐层扩大局部感受野，后续下采样继续扩大其对应的物理空间范围。
            #* 整体分工：时间Attention负责时间演化，2D卷积负责空间关系，多尺度U-Net负责局部到全局。
            queries = encoder(queries)  # 更新预测Query：同一帧相邻BEV位置通过共享的3x3卷积交换信息
            tokens = encoder(tokens)  # 更新上下文Token：使用同一encoder权重建立与Query对齐的空间K/V特征
            # 对tokens的卷积不会引入未来信息：各帧已合入B*F批维，卷积仅在各帧H/W内进行。
            encoder_outs_tokens.append(tokens)  # 保存下采样前的高分辨率Token，解码时恢复上下文空间细节
            encoder_outs_queries.append(queries)  # 保存下采样前的高分辨率Query，解码时恢复预测边界和细节
            #* 2x2、stride=2使H/W各减半、空间Token数约为原来的1/4，并扩大下一尺度的空间感受野。
            # 低分辨率层能够建模更大范围的道路布局、交通流和目标运动关系，而不仅是相邻网格。
            queries = down(queries)  # 预测Query下采样，进入下一空间尺度
            tokens = down(tokens)  # 上下文Token同步下采样，保持下一尺度Q与K/V尺寸和通道对齐
            queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)  # 恢复(B,F,H/2,W/2,C')
            tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f)  # 下一尺度继续执行时间注意力

        #*==================== 5. 最低分辨率瓶颈融合 ====================#
        #* 编码端已完成多尺度时间/空间交互；这里在最低分辨率上进一步融合大范围空间上下文。
        # 下采样后每个特征单元覆盖更大的实际区域，因此瓶颈卷积能以较低计算量整合更远的场景关系。
        b, f, h, w, c = queries.shape  # 记录瓶颈输入形状；默认约为(B,F,12,12,256)
        queries = rearrange(queries, 'b f h w c -> (b f) c h w')  # 合并B/F，转为2D卷积格式
        tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')  # 各帧独立进行空间融合，不跨时间混合
        queries = self.mid(queries)  # 融合预测Query的大范围空间语义，并投影到瓶颈通道数
        tokens = self.mid(tokens)  # 使用同一瓶颈模块处理上下文Token，保持两条场景流的表示对齐

        #* Pose没有H/W维度，无需空间下采样；MLP将其通道投影到与场景瓶颈一致的语义尺度。
        pose_queries = self.pose_mid(pose_queries)  # 更新待预测自车特征的通道表示
        pose_tokens = self.pose_mid(pose_tokens)  # 同步更新已有自车上下文，供解码端继续作为K/V

        #*==================== 6. U-Net解码：上采样、Skip融合并再次执行时空建模 ====================#
        #* 解码端并非简单重复编码端：编码端逐级压缩以获得大范围语义，解码端逐级恢复H/W，并从
        #* Skip Connection取回下采样丢失的边界、位置和小目标细节，最终输出50x50场景Token预测。
        #* Skip拼接改变了当前尺度的Query/Token内容，因此在每个恢复尺度再次执行时间Attention，
        #* 使新取回的高分辨率空间细节也与历史演化对齐；Pose分支也重新读取恢复后的场景特征。
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens, pose_attn_de, pose_de_, encoder_out_pose_queries, encoder_out_pose_tokens, pose_up in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1], self.pose_attn_de, self.pose_de,
                                                                        encoder_outs_pose_queries[::-1], encoder_outs_pose_tokens[::-1], self.pose_up):
            #* 6.1 场景上采样：从低分辨率大范围语义恢复到上一层较高空间分辨率。
            queries = up(queries)  # 预测Query通过2x2反卷积将H/W扩大约2倍，并调整通道数
            tokens = up(tokens)  # 上下文Token同步上采样，保持后续Q与K/V形状一致

            # 当奇数尺寸下采样后无法精确恢复原尺寸时，按编码端Skip尺寸进行对称补边。
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]  # H方向待补长度
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]  # W方向待补长度
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))  # 对齐Query与Skip的H/W
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]  # Token的H方向待补长度
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]  # Token的W方向待补长度
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))  # 对齐Token与Skip的H/W

            #* 6.2 Skip融合：通道拼接“瓶颈传来的大范围语义”与“编码端保留的高分辨率细节”。
            queries = torch.cat([queries, encoder_out_queries], dim=1)  # 拼接预测Query的同尺度编码特征
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)  # 拼接上下文Token的同尺度编码特征
            c, h, w = queries.shape[-3:]  # 记录Skip拼接后的通道数和当前恢复分辨率

            #* 6.3 场景时间重融合：Skip引入的新空间细节尚未在解码尺度完成时间对齐，因此再次沿时间聚合。
            queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)  # 固定位置组成F帧Query序列
            tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)  # 同一位置的历史Token作为K/V
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                # Q=该位置的下一时刻Query，K/V=该位置已有Token；只更新Query，因果mask禁止读取未来。
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f,:f])[0]  # 时间注意力+残差
                queries = cross_norm(queries)  # 对时间注意力结果归一化

                queries = queries + ffn(queries)  # 各位置各时刻独立进行通道非线性变换并残差相加
                queries = ffn_norm(queries)  # FFN结果归一化
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)  # 恢复2D场景Query
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)  # 恢复2D上下文Token

            #* 6.4 Pose Skip融合：将瓶颈自车语义投影回当前通道，并取回编码端同尺度运动细节。
            pose_queries = pose_up(pose_queries)  # 将预测Pose Query投影到当前解码尺度通道数
            pose_tokens = pose_up(pose_tokens)  # 已有Pose Token同步投影，保持Q与K/V维度一致
            pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)  # 拼接编码端Pose Query
            pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)  # 拼接编码端Pose Token

            #* 6.5 Pose再次执行时间聚合并读取当前恢复尺度的场景，使自车预测利用高分辨率环境细节。
            for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_de:
                # Q=下一时刻Pose Query，K/V=已有Pose Token；沿时间读取运动历史，仅更新pose_queries。
                pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]  # Pose时间注意力+残差
                pose_queries = pose_temporal_norm(pose_queries)  # Pose时间注意力结果归一化
                pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')  # 每帧1个Pose Query
                queries = rearrange(queries, '(b f) c h w -> (b f) (h w) c', b=b, f=f, h=h, w=w)  # 每帧H*W个场景Query
                # Q=每帧Pose Query，K/V=同帧全部场景Query；只更新Pose，使其吸收恢复后的空间细节。
                pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]  # Pose-场景空间注意力+残差
                pose_queries = spatial_norm(pose_queries)  # Pose空间注意力结果归一化

                pose_queries = pose_queries + ffn(pose_queries)  # 每个Pose Query独立进行通道变换并残差相加
                pose_queries = ffn_norm(pose_queries)  # Pose FFN结果归一化
                queries = rearrange(queries, '(b f) (h w) c -> (b f) c h w', b=b, f=f, h=h, w=w)  # 恢复2D场景Query
                pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)  # 恢复Pose序列(B,F,C)

            #* 6.6 当前尺度空间细化与通道压缩，为下一次上采样或最终输出做好准备。
            pose_queries = pose_de_(pose_queries)  # MLP融合拼接后的预测Pose特征并压缩通道
            pose_tokens = pose_de_(pose_tokens)  # 同步压缩上下文Pose Token通道
            queries = decoder(queries)  # 3x3 UnetBlock融合相邻位置，并压缩Skip拼接后的Query通道
            tokens = decoder(tokens)  # 使用同一decoder权重细化上下文Token，保持两条场景流对齐

        #*==================== 7. 输出场景码字Logits与下一步自车特征 ====================#
        queries = self.conv_out(queries)  # (B*F,512,H,W)：每个位置对VQ码字的分类logits
        pose_queries = self.pose_out(pose_queries)  # (B,F,128)：交给PoseDecoder预测三模态(dx,dy)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)

        # queries最后一项[:, -1:]预测时间下标mid_frame的场景码字；pose_queries同理。
        return queries, pose_queries  # 与外层z_q_、rel_poses_一一对应，外层取末项并回灌



class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.shortcut = nn.Identity()
            if in_c != out_c:
                self.shortcut = nn.Conv2d(in_c, out_c, 1, 1, 0)
    def forward(self, input):
        output = self.ln(input)
        output = self.conv1(output)
        output = self.act(output)
        output = self.conv2(output)
        if self.residual:
            output = output + self.shortcut(input)
        output = self.act(output)
        return output
