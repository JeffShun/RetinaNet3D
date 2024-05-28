# MedicalObjectDetection3D
## 一、模型介绍
基于RetinaNet3D目标检测框架，网络结构和损失函数根据实际任务做了适当调整。

**注意：本检测模型用于检测3D肝脏，该任务确保每个样本中有且只有一个检测框。本代码并不适用于多目标检测任务，需要自行修改**

## 二、文件结构说明

### 训练文件目录

- train/train.py: 单卡训练代码入口
- train/train_multi_gpu.py: 分布式训练代码入口
- train/custom/dataset/dataset.py: dataset类
- train/custom/model/decoder.py: 预测结果解码器
- train/custom/model/anchor.py: anchor生成
- train/custom/model/neck.py: 模型neck
- train/custom/model/head.py: 模型head
- train/custom/model/loss.py 模型loss
- train/custom/model/backbones/*.py 生成网络backbone
- train/custom/model/network.py: 网络整体框架
- train/custom/utils/*.py 训练相关工具函数
- train/config/model_config.py: 训练的配置文件

### 预测文件目录

* test/test_config.yaml: 预测配置文件
* test/main.py: 预测入口文件
* test/predictor.py: 模型预测具体实现，包括加载模型和后处理
* test/analysis_tools/*.py 结果分析工具函数，如计算评估指标

## 三、demo调用方法

1. 准备训练原始数据
   * 在train文件夹下新建train_data/dcms文件夹，放入用于训练的原始dcm数据
   * 运行 cd train & python custom/utils/dataprepare.py，dcm会被转化为图片存在 train_data/prepared_data目录下
   * 采用labelme等标注工具在train_data/prepared_data下逐切片进行检测框标注，生成json文件
   * 将标注好的数据切分成训练和验证两部分，分别放在train_data/origin_data/train和train_data/origin_data/valid目录下
   * 运行python custom/utils/generate_dataset.py，该命令将数据和标签打包成npz文件，作为网络dataset的输入，此步主要为减少训练时的io操作
   * 运行python custom/utils/anchor_cluster.py，对所有数据的检测框进行kmeans聚类，生成锚框

2. 开始训练
   * 训练相关的参数，全部在train/custom/config/model_config.py 文件中
   * 分布式训练命令：sh ./train_dist.sh
   * 单卡训练命令：python train.py
   * 训练时，每隔指定epoch会输出训练过程中的损失以及验证集指标变化情况，自动保存在train/Logs下，其中 sample/ 目录下存有验证集的检测结果，为.nii.gz格式。tf_logs/ 目录下存有训练的tensorboard日志文件，通过 tensorborad --logdir tf_logs/ --port=6003 进行查看。backup.tar文件为训练时自动备份的当前代码，供有需要时查看和复原，复原命令: python custom/utils/version_checkout.py

3. 准备测试数据
   * 将预测的dcm数据放入test/data/input/dcms目录下，如有labels，将3D检测框label转化为3D mask存在test/data/input/labels目录下，mask的文件格式为.nii.gz

4. 开始预测
   * cd test
   * python main.py

5. 结果评估
   * python test/analysis_tools/cal_matrics.py
