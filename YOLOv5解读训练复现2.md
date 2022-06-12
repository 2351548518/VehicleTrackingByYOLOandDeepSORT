```python
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
```

yolov5.yaml

各个模型区别在于深度因子和宽度因子

```python
[-1, 3, C3, [128]],
# C3的瓶颈层被做的很深
```

上采样和下采样

[(3条消息) 【深度学习基本概念】上采样、下采样、卷积、池化_一朵小冯花的博客-CSDN博客_深度学习下采样率](https://blog.csdn.net/tingzhiyi/article/details/114368433)

```python
# 模型解析
if pretrained:
    with torch_distributed_zero_first(LOCAL_RANK):
        weights = attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
else:
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
amp = check_amp(model)  # check AMP
```

116 train.py

yolo.py

模型文件

```python
class Model(nn.Module):
    # YOLOv5 model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')
```

# Anchor是什么

[(4条消息) Anchor是什么？_"灼灼其华"的博客-CSDN博客_anchor](https://blog.csdn.net/weixin_44285715/article/details/105124650)

```python
# 遍历每一层，backbone
# n backbone的个数
for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
    m = eval(m) if isinstance(m, str) else m  # eval strings
    for j, a in enumerate(args):
        try:
            args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        except NameError:
            pass

    n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
    if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
             BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
        c1, c2 = ch[f], args[0]
        if c2 != no:  # if not output
            c2 = make_divisible(c2 * gw, 8)

        args = [c1, c2, *args[1:]]
        if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x]:
            args.insert(2, n)  # number of repeats
            n = 1
    elif m is nn.BatchNorm2d:
        args = [ch[f]]
    elif m is Concat:
        c2 = sum(ch[x] for x in f)
    elif m is Detect:
        args.append([ch[x] for x in f])
        if isinstance(args[1], int):  # number of anchors
            args[1] = [list(range(args[1] * 2))] * len(f)
    elif m is Contract:
        c2 = ch[f] * args[0] ** 2
    elif m is Expand:
        c2 = ch[f] // args[0] ** 2
    else:
        c2 = ch[f]
```

```python
# 对参数进行额外处理
for j, a in enumerate(args):
    try:
        args[j] = eval(a) if isinstance(a, str) else a  # eval strings
    except NameError:
        pass
```

```py
# max(round(n * gd), 1)  C3中的 n深度 gd 深度参数
n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
```

![image-20220608231907368](https://raw.githubusercontent.com/2351548518/images/main/CCSE/202206082319441.png)

C3层 common.py

```python
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

```python
self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
```

Bottleneck中就有更多的模块
