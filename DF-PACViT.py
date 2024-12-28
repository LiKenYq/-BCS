from torchinfo import summary
import torch.nn.functional as F
from thop import profile
import torch
from torch import nn



class eca_layer(nn.Module):
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.scaling_num = 2

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = 2 * self.sigmoid(y)
        y = y ** self.scaling_num
        x = x * y.expand_as(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, input_channel):
        super(channel_attention, self).__init__()
        self.Average_pool = nn.AdaptiveAvgPool2d(1)
        self.Max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.conv1d = nn.Conv1d(input_channel, input_channel, kernel_size=2, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.bn2 = nn.BatchNorm2d(input_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        Average_out = self.bn1(
            (self.conv(self.Average_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)))
        Max_out = self.bn2((self.conv(self.Max_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)))
        out = torch.cat((Max_out, Average_out), dim=2).squeeze(3)
        out = self.conv1d(out).unsqueeze(3)
        out = self.sigmoid(out)
        return out


class space_attention(nn.Module):
    def __init__(self):
        super(space_attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3,
                              padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        Average_out = torch.mean(x, dim=1, keepdim=True)
        Max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv(torch.cat([Max_out, Average_out], dim=1)))
        return out


class MGUFFM(nn.Module):
    def __init__(self):
        super(MGUFFM, self).__init__()
        self.ca = channel_attention(64, 2)
        self.sa = space_attention()
        self.SiLu = nn.SiLU()
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x1, x2):
        ca_out = self.ca(x2)
        sa_out = self.sa(x2)
        out_att = ca_out * sa_out
        out_att = self.SiLu(out_att)
        x1 = F.interpolate(x1, size=(15, 15), mode='bilinear', align_corners=True)
        mix_out = x1 * out_att
        mix_out = self.bn1(mix_out)
        mix_out = self.SiLu(mix_out)
        return mix_out


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class conv1x1qkv_block(nn.Module):
    def __init__(self, in_planes=64, out_planes=64):
        super(conv1x1qkv_block, self).__init__()
        self.convq = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.convk = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.convv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        q, k, v = self.convq(x), self.convk(x), self.convv(x)
        return q, k, v


class UnfoldBlock(nn.Module):
    def __init__(self, in_planes=64, out_planes=64, head_num=4, kernel_size=5, padding_att=1):
        super(UnfoldBlock, self).__init__()
        self.wise = kernel_size
        self.kernel_size = kernel_size - 4
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=1)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.padding_att = padding_att
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=1)
        self.norm = nn.BatchNorm2d(in_planes).cuda()

    def forward(self, q, k, v):
        b = q.shape[0]
        C = self.in_planes * self.kernel_size * self.kernel_size
        N = ((self.wise - self.kernel_size + 2 * self.padding_att) + 1) * (
                    (self.wise - self.kernel_size + 2 * self.padding_att) + 1)
        pos_embed = nn.Parameter(torch.zeros(1, N, C).cuda())

        cls_tokenq = nn.Parameter(torch.zeros(1, 1, C).cuda())
        cls_tokenq = cls_tokenq.expand(b, -1, -1)

        cls_tokenk = nn.Parameter(torch.zeros(1, 1, C).cuda())
        cls_tokenk = cls_tokenk.expand(b, -1, -1)

        cls_tokenv = nn.Parameter(torch.zeros(1, 1, C).cuda())
        cls_tokenv = cls_tokenv.expand(b, -1, -1)

        q_pad = self.pad_att(q)
        unfold_q = self.unfold(q_pad).permute(0, 2, 1) + pos_embed
        unfold_q = torch.cat([cls_tokenq, unfold_q], dim=1)

        k_pad = self.pad_att(k)
        unfold_k = self.unfold(k_pad).permute(0, 2, 1) + pos_embed
        unfold_k = torch.cat([cls_tokenk, unfold_k], dim=1)

        v_pad = self.pad_att(v)
        unfold_v = self.unfold(v_pad).permute(0, 2, 1) + pos_embed
        unfold_v = torch.cat([cls_tokenv, unfold_v], dim=1)
        return unfold_q, unfold_k, unfold_v


class ConvBlock(nn.Module):
    def __init__(self, out_planes=64, in_planes=64, kernel_conv=5, head_num=4):
        super(ConvBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_conv = kernel_conv
        self.head_num = head_num
        self.head_dim = out_planes // head_num
        self.fc = nn.Conv2d(3 * self.head_num, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False).cuda()
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=0,
                                  stride=1).cuda()
        self.bn = nn.BatchNorm2d(in_planes).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x, q, k, v):
        b, c, h, w = x.shape
        size = h
        f_all = self.fc(torch.cat(
            [q.view(b, self.head_num, self.head_dim, h * w), k.view(b, self.head_num, self.head_dim, h * w),
             v.view(b, self.head_num, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        out_conv = self.dep_conv(f_conv)
        out_conv = self.bn(out_conv)
        out_conv = out_conv.reshape(b, self.in_planes, -1)
        out_conv = out_conv.reshape(b, self.in_planes, size - 4, size - 4)
        return out_conv





class Attention(nn.Module):
    def __init__(self, i, head_num=4, in_planes=64):
        super(Attention, self).__init__()
        self.i = i
        self.unfold = {}
        self.head_num = head_num
        self.head_dim = in_planes // head_num
        self.att_drop = nn.Dropout(0.1)
        self.in_planes = in_planes
        self.convblock1 = ConvBlock()
        self.unfold[1] = UnfoldBlock(kernel_size=19)
        self.unfold[2] = UnfoldBlock(kernel_size=17)
        self.qkv1 = conv1x1qkv_block()
        self.conv_global = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False)

    def forward(self, x1):

        if self.i == 1:
            size = 19
        else:
            size = 17
        q, k, v = torch.tensor([]).cuda(), torch.tensor([]).cuda(), torch.tensor([]).cuda()
        out_conv = torch.tensor([]).cuda()
        q_conv, k_conv, v_conv = self.qkv1(x1)
        q_temp, k_temp, v_temp = self.unfold[self.i](q_conv, k_conv, v_conv)
        q = torch.cat([q, q_temp], dim=2)
        k = torch.cat([k, k_temp], dim=2)
        v = torch.cat([v, v_temp], dim=2)
        conv = self.convblock1(x1, q_conv, k_conv, v_conv)
        out_conv = torch.cat([out_conv, conv], dim=1)
        b, N, C = q.shape
        qurry = q.reshape(b, self.head_num, N, C // self.head_num)
        key = k.reshape(b, self.head_num, N, C // self.head_num)
        value = v.reshape(b, self.head_num, N, C // self.head_num)
        scaling = float(self.head_dim) ** -0.5
        att = (qurry @ key.transpose(-2, -1)) * scaling
        out_attn = (att @ value).transpose(1, 2).reshape(b, N, C)
        out_attn = out_attn[:, 1, :].reshape(b, self.in_planes, size - 4, size - 4)
        out_attn = self.conv_global(out_attn)
        rate1 = torch.nn.Parameter(torch.Tensor(1)).cuda()
        rate2 = torch.nn.Parameter(
            torch.randn(out_attn.shape[0], out_attn.shape[1], out_attn.shape[2], out_attn.shape[3])).cuda()
        init_rate_half(rate1)
        init_rate_half(rate2)
        return out_conv * rate1 + out_attn * rate2


class MPAcV(nn.Module):
    def __init__(self, i):
        super(MPAcV, self).__init__()
        self.i = i
        self.Attention = Attention(self.i)

    def forward(self, x1):
        # print(self.i)
        out = self.Attention(x1)
        return out


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, i, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = MPAcV(i)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out





class HyperBCS_2D(nn.Module):
    def __init__(self, block=Bottleneck, input_channels=28, num_classes=11, groups=1,norm_layer=None):
        super(HyperBCS_2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=50, out_channels=self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool_multiscale = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.layer1 = Bottleneck(inplanes=64, planes=64, i=1)
        self.layer2 = Bottleneck(inplanes=64, planes=64, i=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.scale1_rate = torch.nn.Parameter(torch.Tensor(1)).cuda()
        self.scale2_rate = torch.nn.Parameter(torch.Tensor(1)).cuda()
        self.conv_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.ECA = eca_layer(64)
        self.MGUFFM = MGUFFM()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv2_yd = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.scale1_rate)
        init_rate_half(self.scale2_rate)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.ECA(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2_yd = self.conv2_yd(x2)
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x_MGUFFM = self.MGUFFM(x2, x2_yd)
        x_total = torch.cat((x1,x_MGUFFM),dim=1)
        x_total = self.conv_fusion(x_total)
        x_total = self.relu(x_total)
        x_total = self.avgpool(x_total)
        x_total = torch.flatten(x_total, 1)
        x_total = self.fc(x_total)
        return x_total


if __name__ == '__main__':
    model = HyperBCS_2D().cuda()
    input = torch.randn([32, 35, 9, 9]).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(model(input).shape)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}M".format(flops / 1e6))
    print("params:{:.3f}M".format(params / 1e6))
    summary(model, input_size=(32, 50, 9, 9))
