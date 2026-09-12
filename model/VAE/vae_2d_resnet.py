""" adopted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py """
# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import numpy as np
from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch.nn.functional as F
from copy import deepcopy

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
    
    def forward(self, x, shape):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        diffY = shape[0] - x.size()[2]
        diffX = shape[1] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])

        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
    
    def forward(self, x):
        if self.with_conv:
            #pad = (0, 1, 0, 1, 0, 1)
            #x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x
    


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_
    
@MODELS.register_module()
class VAERes2D(BaseModule):
    def __init__(
            self, 
            encoder_cfg, 
            decoder_cfg,
            num_classes=18,
            expansion=8, 
            vqvae_cfg=None,
            init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion
        self.num_cls = num_classes

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)
        self.class_embeds = nn.Embedding(num_classes, expansion)

        if vqvae_cfg:
            self.vqvae = MODELS.build(vqvae_cfg)
        self.use_vq = vqvae_cfg is not None
    
    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        # x: (B,F,H,W,D)，每个体素保存一个整数语义类别编号，而不是连续特征。
        bs, F, H, W, D = x.shape  # 读取批大小、帧数、BEV长宽和高度层数
        x = self.class_embeds(x)  # 类别编号->可学习向量：(B,F,H,W,D)->(B,F,H,W,D,C_emb)
        # 后续 Encoder2D 只接收 (N,C,H,W)，所以需要将每个 BEV 位置上 D 个高度层的
        # C_emb 维语义向量依高度顺序拼接为一个 D*C_emb 维“垂直柱”特征。
        # 这里是拼接而非求和/平均：第 d 层始终占据固定的一段通道，因此没有直接抹去高度顺序；
        # 它以通道形式保存三维信息，从而用计算成本更低的 2D CNN 代替 3D CNN。
        x = x.reshape(
            bs*F, H, W, D * self.expansion).permute(
                0, 3, 1, 2)  # (B,F,H,W,D,C_emb)->(B*F,D*C_emb,H,W)，F作为独立帧编码

        z, shapes = self.encoder(x)  # 2D卷积编码并降采样；z为连续潜特征，shapes记录中间空间尺寸
        return z, shapes  # z交给VQ码本离散化，shapes供decoder逐级恢复分辨率
        
    def forward_decoder(self, z, shapes, input_shape):
        logits = self.decoder(z, shapes)

        bs, F, H, W, D = input_shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0) # 1, expansion, cls
        similarity = torch.matmul(logits, template) # -1, D, cls
        # pred = similarity.argmax(dim=-1) # -1, D
        # pred = pred.reshape(bs, F, H, W, D)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)

    def forward(self, x, **kwargs):
        # xs = self.forward_encoder(x)
        # logits = self.forward_decoder(xs)
        # return logits, xs[-1]
        
        output_dict = {}
        z, shapes = self.forward_encoder(x)
        if self.use_vq:
            z_sampled, loss, info = self.vqvae(z, is_voxel=False)
            output_dict.update({'embed_loss': loss})
        else:
            z_sampled, z_mu, z_sigma = self.sample_z(z)
            output_dict.update({
                'z_mu': z_mu,
                'z_sigma': z_sigma})
        
        logits = self.forward_decoder(z_sampled, shapes, x.shape)
        
        output_dict.update({'logits': logits})
    
        if not self.training:
            pred = logits.argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)
            
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            
        return output_dict
        # loss, kl, rec = self.loss(logits, x, z_mu, z_sigma)
        # return loss, kl, rec
        
    def generate(self, z, shapes, input_shape):
        logits = self.forward_decoder(z, shapes, input_shape)
        return {'logits': logits}



@MODELS.register_module()
class Encoder2D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Enc has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        # ==================== Encoder2D：BEV场景特征压缩 ====================
        # 整体作用：把高分辨率BEV特征编码成低分辨率连续潜特征，随后再由VQ码本
        # 将每个低分辨率位置离散化成一个场景token。本模块只做空间特征编码，
        # 不建模帧间关系，因为时间维F已并入batch，各帧在这里相互独立地通过同一编码器。
        # 默认形状变化：
        #   原始occupancy             (B,F,200,200,16)
        #   类别嵌入并折叠高度后      (B*F,128,200,200)  <- 本函数输入x
        #   两次2倍下采样后           (B*F,128, 50, 50)  <- 本函数输出h
        # 因此H/W分别压缩4倍，BEV位置数压缩16倍；通道保存局部三维语义结构。
        # x: (B*F,D*C_emb,H,W)，高度特征已按层拼接进通道维。
        shapes = []  # 记录各次下采样前的二维尺寸，供Decoder2D恢复对应分辨率
        temb = None  # 当前模型不使用扩散模型式的时间步嵌入，但保留ResnetBlock接口

        h = self.conv_in(x)  # 3x3卷积：将D*C_emb个输入通道投影到基础通道数ch
        for i_level in range(self.num_resolutions):  # 依次处理由高到低的多个空间尺度
            
            for i_block in range(self.num_res_blocks):  # 当前尺度连续执行多个残差块
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)  # 提取局部空间特征并调整通道数

                if len(self.down[i_level].attn) > 0:  # 配置要求当前分辨率使用空间自注意力时
                    h = self.down[i_level].attn[i_block](h)  # 建模相距较远BEV位置间的全局关系
                # hs.append(h)
            if i_level != self.num_resolutions-1:  # 最低分辨率层之后不再继续下采样
                shapes.append(h.shape[-2:])  # 保存下采样前的(H,W)，当前解码器接口仍接收该列表
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)  # 每次H/W减半；两次后200->100->50，累计压缩4倍

        # 瓶颈层：在最低空间分辨率上进一步融合局部特征和全局上下文。
        # h = hs[-1]
        #
        h = self.mid.block_1(h, temb)  # 第一个瓶颈残差块
        h = self.mid.attn_1(h)  # 最低分辨率自注意力，以较低成本建立全局空间联系
        h = self.mid.block_2(h, temb)  # 第二个瓶颈残差块，继续融合注意力输出

        # 输出头：规范化和激活后，将通道投影成VQ量化器所需的潜特征维度。
        h = self.norm_out(h)  # GroupNorm稳定不同通道的特征分布
        h = nonlinearity(h)  # Swish非线性激活
        h = self.conv_out(h)  # 输出连续潜特征z：(B*F,z_channels,H/4,W/4)，默认(B*F,128,50,50)
        return h, shapes  # h后续进入quant_conv和码本量化；shapes传给decoder

@MODELS.register_module()
class Decoder2D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align with encoder
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, shapes):
        # ==================== Decoder2D：BEV潜特征重建 ====================
        # 整体作用：把VQ码字经过post_quant_conv后的低分辨率潜特征逐级上采样，
        # 恢复为高分辨率BEV“垂直柱”特征；随后VAERes2D.forward_decoder()再把
        # D*C_emb个输出通道拆回D个高度层，并计算每个体素的语义类别logits。
        # 默认形状变化：(B*F,128,50,50) -> (B*F,128,200,200)。
        # z: (B*F,C,H/d,W/d)，默认(B*F,128,50,50)，尚不是最终occupancy类别。
        self.last_z_shape = z.shape  # 保存本次潜特征形状，便于调试或外部查询

        temb = None  # 当前模型不使用扩散时间步嵌入，仅为兼容ResnetBlock接口

        h = self.conv_in(z)  # 3x3卷积：将z_channels投影到解码器最低分辨率的通道数

        # 瓶颈层先在50x50低分辨率上融合特征；此处分辨率低，执行注意力成本较小。
        h = self.mid.block_1(h, temb)  # 第一个瓶颈残差块，提取局部结构
        h = self.mid.attn_1(h)  # 空间自注意力，建立远距离BEV位置之间的关系
        h = self.mid.block_2(h, temb)  # 第二个瓶颈残差块，进一步融合注意力输出

        for i_level in reversed(range(self.num_resolutions)):  # 从最低尺度逐级恢复到原始BEV尺度
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks):  # 每个尺度使用与Encoder2D数量对应的残差块
                h = self.up[i_level].block[i_block](h, temb)  # 重建当前尺度局部特征并调整通道数
                if len(self.up[i_level].attn) > 0:  # 配置指定当前分辨率使用注意力时
                    h = self.up[i_level].attn[i_block](h)  # 补充当前尺度的全局空间关系
            if i_level != 0:  # 最高分辨率层不再继续上采样
                h = self.up[i_level].upsample(
                    h, shapes.pop())  # H/W约扩大2倍；用Encoder记录尺寸精确恢复50->100->200

        if self.give_pre_end:  # 可选：直接返回输出头之前的高分辨率隐藏特征
            return h  # 跳过归一化、激活和最终通道投影

        h = self.norm_out(h)  # GroupNorm稳定高分辨率重建特征的通道分布
        h = nonlinearity(h)  # Swish非线性激活
        h = self.conv_out(h)  # 输出(B*F,D*C_emb,H,W)，默认(B*F,128,200,200)
        return h  # 返回BEV垂直柱特征；外层函数继续拆分高度并生成18类体素logits


if __name__ == "__main__":
    # test encoder
    import torch
    encoder = Encoder2D(in_channels=3, ch=64, out_ch=64, ch_mult=(1,2,4,8), num_res_blocks=2, resolution=200,attn_resolutions=(100,50), z_channels=64, double_z=True)
    #decoder = Decoder3D()
    decoder = Decoder2D(in_channels=3, ch=64, out_ch=3, ch_mult=(1,2,4,8), num_res_blocks=2, resolution=200,attn_resolutions=(100,50), z_channels=64, give_pre_end=False)
    
    import pdb; pdb.set_trace()
