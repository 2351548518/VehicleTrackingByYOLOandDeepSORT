```python
train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
```

数据处理的类，train.py 221行

![image-20220605192625543](https://raw.githubusercontent.com/2351548518/images/main/CCSE/202206051926655.png)

数据增强参数

```python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Hyperparameters for high-augmentation COCO training from scratch
# python train.py --batch 32 --cfg yolov5m6.yaml --weights '' --data coco.yaml --img 1280 --epochs 300
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.3  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 0.7  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.9  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.1  # image mixup (probability)
copy_paste: 0.1  # segment copy-paste (probability)

```

## 2源码数据

```python
# augment=False 是否使用数据增强
def __init__(self,
             path,
             img_size=640,
             batch_size=16,
             augment=False,
             hyp=None,
             rect=False,
             image_weights=False,
             cache_images=False,
             single_cls=False,
             stride=32,
             pad=0.0,
             prefix=''):
    self.img_size = img_size
    self.augment = augment
    self.hyp = hyp
    self.image_weights = image_weights
    self.rect = False if image_weights else rect
    self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
    self.mosaic_border = [-img_size // 2, -img_size // 2]
    self.stride = stride
    self.path = path
    self.albumentations = Albumentations() if augment else None
```

```python
# 获取图片路径地址
try:
    f = []  # image files
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            # f = list(p.rglob('*.*'))  # pathlib
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{prefix}{p} does not exist')
    self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
    # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
    assert self.im_files, f'{prefix}No images found'
except Exception as e:
    raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')
```

```python
# Check cache
# 图片和对应的标签文件解析并保存到本地
self.label_files = img2label_paths(self.im_files)  # labels
cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
try:
    cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
    assert cache['version'] == self.cache_version  # same version
    assert cache['hash'] == get_hash(self.label_files + self.im_files)  # same hash
except Exception:
    cache, exists = self.cache_labels(cache_path, prefix), False  # cache
```

```python
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
```

```python
def cache_labels(self, path=Path('./labels.cache'), prefix=''):
    # Cache dataset labels, check images and read shapes
    x = {}  # dict
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
    with Pool(NUM_THREADS) as pool:
        pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                    desc=desc,
                    total=len(self.im_files),
                    bar_format=BAR_FORMAT)
        for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if im_file:
                x[im_file] = [lb, shape, segments]
            if msg:
                msgs.append(msg)
            pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

    pbar.close()
    if msgs:
        LOGGER.info('\n'.join(msgs))
    if nf == 0:
        LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
    x['hash'] = get_hash(self.label_files + self.im_files)
    x['results'] = nf, nm, ne, nc, len(self.im_files)
    x['msgs'] = msgs  # warnings
    x['version'] = self.cache_version  # cache version
    try:
        np.save(path, x)  # save cache for next time
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {path}')
    except Exception as e:
        LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
    return x
```

```python
def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    # 对JPEG格式图片的角度进行修正
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]
```

```python
# Update labels
include_class = []  # filter labels to include only these classes (optional)
include_class_array = np.array(include_class).reshape(1, -1)
for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
    if include_class:
        j = (label[:, 0:1] == include_class_array).any(1)
        self.labels[i] = label[j]
        if segment:
            self.segments[i] = segment[j]
    if single_cls:  # single-class training, merge all classes into 0
        self.labels[i][:, 0] = 0
        if segment:
            self.segments[i][:, 0] = 0
```

```python
# 是否采用矩形训练方式 保证图片缩放后黑边最少
if self.rect:
    # Sort by aspect ratio
    s = self.shapes  # wh
    # 高和宽的比例
    ar = s[:, 1] / s[:, 0]  # aspect ratio
    irect = ar.argsort()
    self.im_files = [self.im_files[i] for i in irect]
    self.label_files = [self.label_files[i] for i in irect]
    self.labels = [self.labels[i] for i in irect]
    self.shapes = s[irect]  # wh
    ar = ar[irect]

    # Set training image shapes
    shapes = [[1, 1]] * nb
    for i in range(nb):
        ari = ar[bi == i]
        mini, maxi = ari.min(), ari.max()
        if maxi < 1:
            shapes[i] = [maxi, 1]
        elif mini > 1:
            shapes[i] = [1, 1 / mini]

    self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
```

```python
def __getitem__(self, index):
    index = self.indices[index]  # linear, shuffled, or image_weights

    hyp = self.hyp
    mosaic = self.mosaic and random.random() < hyp['mosaic']
    if mosaic:
        # Load mosaic
        img, labels = self.load_mosaic(index)
        shapes = None

        # MixUp augmentation
        if random.random() < hyp['mixup']:
            img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

    else:
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

    nl = len(labels)  # number of labels
    if nl:
        labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

    if self.augment:
        # Albumentations
        img, labels = self.albumentations(img, labels)
        nl = len(labels)  # update after albumentations

        # HSV color-space
        augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        # Flip up-down
        if random.random() < hyp['flipud']:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if random.random() < hyp['fliplr']:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        # Cutouts
        # labels = cutout(img, labels, p=0.5)
        # nl = len(labels)  # update after cutout

    labels_out = torch.zeros((nl, 6))
    if nl:
        labels_out[:, 1:] = torch.from_numpy(labels)

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    # 返回图片和标签
    return torch.from_numpy(img), labels_out, self.im_files[index], shapes
```
