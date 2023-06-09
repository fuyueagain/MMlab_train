# MMPretrain 微调task
文档说明：
- 模型参数设置myconfig：[resnet50_fruit30.py](https://github.com/fuyueagain/MMlab_train/blob/main/02_MMPretrain/resnet50_fruit30.py)
- train训练日志：[20230608_092805.log](https://github.com/fuyueagain/MMlab_train/blob/main/02_MMPretrain/20230608_092805.log)
- test训练日志：[20230608_093427.log](https://github.com/fuyueagain/MMlab_train/blob/main/02_MMPretrain/20230608_093427.log)
![test](https://github.com/fuyueagain/MMlab_train/blob/main/02_MMPretrain/test_res.jpg)
- 结果图片：预测的效果不错👍
![xigua](https://github.com/fuyueagain/MMlab_train/blob/main/02_MMPretrain/xigua_res.jpg)
![youzi1](https://github.com/fuyueagain/MMlab_train/blob/main/02_MMPretrain/youzi_red.png)
![youzi2](https://github.com/fuyueagain/MMlab_train/blob/main/02_MMPretrain/youzi_res.jpg)

## 1. 加载数据分割数据集
``` shell
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain/
pip install openmin
min install -e ".[multimodal]" 
# 加载数据集
cd data
tar -xf fruit30_train.tar 
```
> 训练集和验证集比例为8:2,代码如下：

``` python
import os
import shutil
import random

# 定义数据集路径和保存路径
data_dir = 'fruit30_train'
new_data_dir = 'fruit30_dataset'
os.makedirs(new_data_dir, exist_ok=True)

train_dir = os.path.join(new_data_dir, 'training_set')
val_dir = os.path.join(new_data_dir, 'val_set')

# 定义训练集和验证集的比例
train_ratio = 0.8
val_ratio = 0.2

# 创建保存路径
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取数据集中所有子文件夹的名称
subdirs = os.listdir(data_dir)

# 遍历所有子文件夹
for subdir in subdirs:
    subdir_path = os.path.join(data_dir, subdir)
    if os.path.isdir(subdir_path):
        # 获取当前子文件夹中所有图像的文件名
        images = os.listdir(subdir_path)
        # 随机打乱图像的顺序
        random.shuffle(images)
        # 计算训练集和验证集的大小
        num_images = len(images)
        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)
        # 将图像按比例划分为训练集和验证集
        train_images = images[:num_train]
        val_images = images[num_train:num_train+num_val]
        # 将划分好的训练集和验证集保存到对应的文件夹中
        
        for image in train_images:
            src_path = os.path.join(subdir_path, image)
            dst_path = os.path.join(train_dir, subdir, image)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
        for image in val_images:
            src_path = os.path.join(subdir_path, image)
            dst_path = os.path.join(val_dir, subdir, image)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
print('complete！！！')
```

## 2. 配置训练设置并开始训练验证
```bash
# 设置模型myconfigs文件夹，并上传resnet50_fruit30.py
mkdir myconfigs

# 开始训练，先回到mmpretrain主文件夹，两种方法都可
mim train mmpretrain myconfigs/resnet50_fruit30.py --work-dir=work_dirs
# python tools/train.py \myconfigs/resnet50_fruit30.py --work-dir=work_dirs

# 打开work_dirs找到保留的最好的checkpoint，并验证
mim test mmpretrain myconfigs/resnet50_fruit30.py --checkpoint work_dirs/best_accuracy_top1_epoch_9.pth
```

## 3. 使用自己的水果图片预测
可以使用ImageClassificationInferencer这个API推理，或者demo/image_demo.py推理
- API推理
```python
import os
# os.chdir('mmpretrain')
os.getcwd()

# API推理
from mmpretrain import ImageClassificationInferencer
inf=ImageClassificationInferencer('myconfigs/resnet50_fruit30.py',pretrained='work_dirs/best_accuracy_top1_epoch_9.pth')
img_path='data/inputs/xigua_red.jpg'
res=inf(img_path,show=True)
print(res)
# 输出结果
pred_label=res[0]['pred_label']
pred_score=res[0]['pred_score']
pred_class=res[0]['pred_class']
print(pred_label,pred_score,pred_class)
# 绘制输出图片
# 可视化
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

img2=cv2.imread(img_path)
# cv2.putText(img2, u'label:{} socre:{:.2f} class:{}'.format(pred_label,pred_score,pred_class), (100, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA,fontFile=font_file)
img = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img)
fontText = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 30, encoding="utf-8")
draw.text((50,50 ), 'label:{} socre:{:.2f} class:{}'.format(pred_label,pred_score,pred_class), (255,255,255), font=fontText)

img = np.asarray(img)
img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

new_path=img_path.split('/')[-1]
cv2.imwrite(f'outputs/{new_path}',img)

plt.imshow(img)
```
- demo/image_demo.py推理
```bash
python \
    demo/image_demo.py \
    data/inputs/youzi_red.jpg \
    myconfigs/resnet50_fruit30.py \
    --checkpoint work_dirs/best_accuracy_top1_epoch_9.pth \
    --show-dir outputs/
```

```python
# 查看预测出来的图片
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('outputs/youzi_red.png')

plt.imshow(img)

plt.show()
```

字体设置
```bash
sudo apt-get install fonts-wqy-zenhei
```

```python
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", size=12)
```


其他方法 [参考](https://sujingjhong.com/posts/how-to-show-matplotlib-visual-packages-in-chinese-on-colab/)
```python
# Show Chinese in matplotlib

# Show Chinese
!wget 'https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKtc-hinted.zip'
!mkdir /tmp/fonts
!unzip -o NotoSansCJKtc-hinted.zip -d /tmp/fonts/
!mv /tmp/fonts/NotoSansMonoCJKtc-Regular.otf /usr/share/fonts/truetype/NotoSansMonoCJKtc-Regular.otf -f
!rm -rf /tmp/fonts
!rm NotoSansCJKtc-hinted.zip
# 指定字體
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

font_dirs = ['/usr/share/fonts/truetype/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
  font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = "Noto Sans Mono CJK TC"
```
