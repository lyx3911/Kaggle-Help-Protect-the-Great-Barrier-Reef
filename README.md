# Kaggle-Help-Protect-the-Great-Barrier-Reef

kaggle上的一个奖金赛：ensorFlow - Help Protect the Great Barrier Reef (Detect crown-of-thorns starfish in underwater image data) 

详细介绍：[TensorFlow - Help Protect the Great Barrier Reef | Kaggle](https://www.kaggle.com/c/tensorflow-great-barrier-reef/overview)

## 赛事简介

这项比赛的目标是通过珊瑚礁的水下视频建立目标检测模型，实时准确地识别海星。

数据为若干段视频，共23502帧视频，但其中大多数帧都没有待检测的目标。

参加这个比赛的原因是一来寒假没事做，二来觉得这个比赛和我之前调研过的Video Object Detection比较相关。

## 模型和数据集

模型：yolov5s (预训练的参数)

数据集：kaggle提供的训练集，未使用额外数据。

评估指标：F2-score

## 硬件和软件

设备：2080Ti

系统：Ubuntu 18.0

深度学习框架：Pytorch1.8

## 方法

#### 过程

题外话：这个比赛是拿不到测试数据的，需要用官方提供的数据接口写好自己的推理代码，并且提交的notebook需要能够不联网运行，这就导致配环境非常复杂。本来我是想用MMDetection的，但是在Kaggle上配置能不联网运行的MMDetection环境真的太折磨了，虽然最后参考大佬的教程配好了，但是每次调试都要重新编译mmcv-full真的太难受了。所以最后还是选择了更加方便的yolov5。

##### 数据预处理：

训练数据一共三段长视频，我删除了训练图片中没有待检测目标的帧，剩下的帧大约属于20多个小视频片段，将这些片段按10fold划分了训练集和验证集，没有直接随机划分的原因是视频相邻两帧的差异比较小，如果随机划分会造成验证集泄露。

将标签处理成yolov5的格式。

- **baseline：**yolov5s训练上述处理后的数据，所有的参数都设为默认值，仅修改image size为1280（因为图片的尺寸为1920*720）score：0.45
- **更复杂的模型：**尝试yolo5l，score：0.49，本来想尝试yolo5x，但是显存不够，size为1280时单卡batchsize就只能设为1了，即使使用了四卡并行效率也很低，遂放弃。
- **更大的image size：**尝试在验证的时候增大image size，发现在验证集上的score有所提升。于是尝试训练时也使用更大的image size，随后尝试下来训练的时候image size设为2560，推理的时候image size设为4800的效果是最好的。（在kaggle讨论区看到很多High resolution is all you need的帖子，他们甚至尝试了3600的image size来训练，image size=10000来推理，我只能说显存大了不起啊。不过线下尝试的时候更大的image size并没有起作用，猜测可能他们做数据增强的时候缩放比例更大，也有可能是我batchszie太小）score：0.6左右
- **Test-Time Augmentation：**推理时进行数据增强，然后对增强后的预测结果进行融合，为yolov5自带的功能。能稍微提升一点点分数。
- **数据增强：**调整了一下默认的数据增强方式，加强了HSV变换，扩大了缩放系数，mixup和cutpaste分别设为0.5。
- **调整置信度阈值：**默认的阈值为0.6，太高了。因为海星是比较容易和背景混淆的目标，观察模型的输出发现有很多置信度只有0.3左右的检测框其实是正确的。同时这个比赛的评估函数是F2-score，所以召回率更加重要。修改了yolov5中的f1-score计算函数，在log文件中记录每一轮的f2-score。模型训练完成后，观察在验证集上的f2-score曲线，选择最好的置信度阈值。阈值设为0.45的时候，分数可以达到0.65左右。
- **Tracking：**因为是视频信息，所以将前后帧关联起来一定比单看一帧图像更好，所以使用一个简单的目标跟踪器跟踪每一个海星，并预测它在下一帧会出现的位置和概率，与下一帧的目标检测结果进行融合，具体的实现借助了[norfair](https://github.com/tryolabs/norfair)库，所以实现起来并不复杂。

#### 最终方案

模型：yolov5s，使用预训练的模型初始化。

训练时的image size为2560，推理时的size为4800，并进行Test-Time Augmentation，置信度阈值为0.45。

使用目标跟踪算法结合前面几帧的检测结果对当前帧进行修正。

最终提交的notebook：`yolov5-tracking.ipynb`

#### 其他方案

尝试了一些我觉得可能会有用的方法，但是分数并没有提高：

- **Model Ensemble：**10折交叉验证，选取其中三折训练的模型进行WBF融合（目标框加权融合），利用ensemble_boxes PyPI实现，提交过一次效果并不好，后面也没有时间训练更多的模型用于调试，所以就放弃了。
- 加入一部分没有目标的训练集：按照yolov5官方文档中推荐的，加了10%，没什么用。
- **滑窗：**在较大的图片中，对小目标的
- **soft-NMS：**将yolov5的后处理模块NMS改成soft-NMS，和不改没什么区别
- **tph-yolov5：**在其他比赛的经验帖中看到的，给yolov5加了基于transformer结构的小目标检测头，具体可见我魔改过的yolov5：`yolov5/models/common.py`。用这个改过的yolov5s比不改的更差，可能是因为本身训练数据也不是很多，tph-yolov5没有预训练的模型。（yolov5s随机初始化要比使用预训练模型低0.08左右）
- Video Anomaly Detection领域用到的方法
  - seq-NMS：
- 伪装目标检测

## 最终成绩

银牌 top4%

A榜：0.679

B榜：0.683

## 感想

## TOP方案总结

// TODO

## Reference
