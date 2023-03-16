# 级联金字塔网络的Pytorch实现，用于时尚AI的关键点检测

<div align="center">

[English](README_original.md) | 简体中文

</div>

这段代码在Pytorch中实现了用于多人姿势估计的级联金字塔网络，以检测服装的关键点，包括五种类型：上衣、礼服、外衣、裙子和裤子。这是在正式代码发布之前开始的，所以有一些差异。在实验中，测试了对CPN的各种修改，其中ResNet-152主干网和SENet-154主干网显示了更好的结果。

<div align="center">
<img src="demo/blouse.png" width="256"/> <img src="demo/dress.png" width="256"/> <img src="demo/outwear.png" width="256"/> <img src="demo/skirt.png" width="256"/> <img src="demo/trousers.png" width="256"/>
<p> Example output of CPN with ResNet-152 backbone </p>
</div>

## 开始

该中文版本的作者并非原作者。本文作者在复现2018年的原始版本过程中发现，原始版本并不适用于Python>=3.7（经测试发现对Python3.6兼容），因此对原始版本做出了略微修改和注释。本文档说明写于2023年2月28日，并非完全是对原文档的翻译，还有作者对原文档做出的补充和基于最近commit的修改。

The author of the Chinese version is not the original author. In the process of reproducing the 2018 original, the author found that the original does not work with Python>=3.7 (it was tested and found to be compatible with Python3.6), so minor changes were made and commented to the original. This documentation note, written on February 28, 2023, is not a complete translation of the original document, but contains additions to the original document and changes made by the author based on the recent commit.

### 环境

- Python >=3.7

- Numpy
- Pandas
- PyTorch
- cv2
- scikit-learn
- py3nvml and nvidia-ml-py3
- tqdm
- random
- math

### 数据准备

本版本的数据集来源（应该是2018年参赛的人后来上传的，原始路径不存在了）：[FashionAI dataset](https://tianchi.aliyun.com/dataset/136923)。数据集描述如下：

|数据名称|大小| 描述  |
|----|----|-----|
|README.md|9.56KB| 数据文档 |
|eval.zip|1.17KB|     |
|fashionAI_keypoints_train2.tar|2.10GB| 训练集2 |
|fashionAI_keypoints_train1.tar|3.00GB| 训练集1 |
|fashionAI_keypoints_test.tar|3.00GB| 测试集 |
|FashionAI_A_Hierarchical_Dataset_<br/>for_Fashion_Understanding|674.88KB| FashionAI数据集论文 |

⚠️在此版本中，数据集的路径与原始版本并不相同，且仅使用了原始数据集的5%。文件路径如下

```
fashion/
  |-- checkpoints
  |-- tmp
  |    |-- one
  |    |-- ensemble
  |-- kp_predictions
  |    |-- one
  |    |-- ensemble
  |-- KPDA/
       |-- test_extracted.csv
       |-- train_extracted.csv
       |-- train1/
       |    |-- blouse/
       |    |-- dress/
       |    |-- outwear/
       |-- train2/
       |    |-- skirt/
       |    |-- trousers/
       |-- test/
            |-- blouse/
            |-- dress/
            |-- outwear/
            |-- skirt/
            |-- trousers/
            |-- test.csv
```

`fasion/`是本版本用到的数据集的根目录

- train1->fashionAI_keypoints_train1.tar
- train2->fashionAI_keypoints_train2.tar
- test->fashionAI_keypoints_test.tar

在`config.py`中，可以通过修改`proj_path`来修改数据路径，包括数据读取路径和checkpoints、运行结果的保存路径

## 模型训练

**超参数（`batch size`, `cuda devices`, `learning rate` ,`workers`,`epoch`）在`config.py`中进行修改**

### 从零开始训练模型

`python3 src/stage2/trainval.py -c {clothing type}`

或

`python src/stage2/trainval.py -c {clothing type}`

使用`-c`或者`--clothes`来选择服装类型（`blouse`,`dress`,`outwear`,`skirt`,`trouser`中的一种）。

你也可以通过下列代码来自动运行

`bash src/stage2/autorun.sh`

它实际上为五种服装类型运行了`stage2/trainval.py`五次。

### 从checkpoints恢复训练

> [理解Checkpoint - 知乎](https://zhuanlan.zhihu.com/p/410548507#:~:text=Checkpoi,%EF%BC%89%E7%9A%84%E6%83%AF%E4%BE%8B%E6%88%96%E6%9C%AF%E8%AF%AD%E3%80%82)

`python3 src/stage2/trainval.py -c {clothing type} -r {path_to_the_checkpoint}`

或

`python src/stage2/trainval.py -c {clothing type} -r {path_to_the_checkpoint}`

当恢复训练时，步数、学习率和优化器状态也将从checkpoints恢复。对于SGD优化器，优化器状态包含每个可训练参数的动量（[momentum](https://blog.csdn.net/gaoxueyi551/article/details/105238182)）。例如（代码见`trainval.py` `line 187 to 193`）：

```python
torch.save({
    'epoch': epoch,
    'save_dir': save_dir,
    'state_dict': state_dict,
    'lr': lr,
    'best_val_loss': best_val_loss},
    os.path.join(save_dir, 'kpt_' + config.clothes + '_%03d.ckpt' % epoch))
```

### 训练脚本的背后

- 数据预处理在`stage2/data_generator.py`中进行，在训练中调用。

- 本次挑战赛使用了两个网络，分别是` stage2/cascaded_pyramid_network.py` 和 `stage2v9/cascaded_pyramid_network_v9.py`。最后的分数来自于集合学习。这两个网络共享相同的架构，但骨架不同。

- 所有其他版本都是失败的实验，可以暂时忽略。

## 模型验证和测试

### 基于验证集的模型验证

为了验证模型，请运行下列代码：

`python3 src/stage2/predict_one.py -c {clothing type} -g {gpu index} -m {path/to/the/model} -v {True/False}`

为了验证两个模型的综合性能，请运行：

`python3 src/stage2/predict_ensemble.py -c {clothing type} -g {gpu index} -m1 {path/to/the/model1} -m2 {path/to/the/model2} -v {True/False}`

在程序结束时，会打印出`normalized error`

### 在测试集上生成用于成绩提交的结果

测试单个模型，请运行：

`python3 src/kpdetector/predict.py -c {clothing type} -g {gpu index} -m {path/to/the/model} -v {True/False}`

测试两个模型的综合性能：

`python3 src/kpdetector/predict_ensemble.py -c {clothing type} -g {gpu index} -m1 {path/to/the/model1} -m2 {path/to/the/model2} -v {True/False}`

运行`python3 src/kpdetector/concatenate_results.py`以此将所有用于提交的`.csv`文件进行合并

## 实验（`normalized error`的降低）

- Replace ResNet50 by ResNet152 as backbone network (-0.5%)
- Increase input resolution from 256x256 to 512x512 (-2.5%)
- Gaussian blur on predicted heatmap (-0.5%)
- Reduce rotaton angle from 40 degree to 30 for data augmentation (-0.6%)
- Use (x+2, y+2) where (x, y) is max value coordinate (-0.4%)
- Use 1/4 offset from coordinate of the max value to the one of second max value (-0.2%)
- Flip left to right for data augmentation (-0.2%)

## Benchmark

This solution achieved LB 3.82% in Tianchi FashionAI Global Challenge, 17th place out 2322 teams. Check the leaderboard [here](https://tianchi.aliyun.com/competition/entrance/231648/rankingList).
