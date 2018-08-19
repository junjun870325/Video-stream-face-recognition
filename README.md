# Video-stream-face-recognition

前言：

本文开发的是一个完整的视频流人脸识别系统，主要流程如下：

首先，通过openCV抓取摄像头的视频流

第二，通过MTCNN对每帧图片进行人脸检测和对齐，当然考虑到效率我们可以每n个间隔帧进行一次检测

第三，通过facenet预训练模型对第二步得到的人脸进行512维的特征值提取

第四，收集目标数据集来训练自己的分类模型

第五，将第三部得到的512维的特征值作为第四部的输入，然后输出即为我们类别值

准备工作：

安装openCV

pip3 install opencv-python

下载facenet，其中src/align下是MTCNN的tensorflow实现及预训练模型

git clone --recursive https://github.com/davidsandberg/facenet.git

LFW数据集下载地址：http://vis-www.cs.umass.edu/lfw/

通过以下命令对LFW数据进行人脸检测和对齐，这里我们获取160*160大小的图像以备后面使用，如果你有自己的数据集，可以忽略，先设置环境变量

export PYTHONPATH=/Users/admin/facenet/src
for N in {1..4}; do python3 /Users/admin/facenet/src/align/align_dataset_mtcnn.py /Users/admin/lfw /Users/admin/lfw_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done

models目录下的模型文件请到以下地址下载

https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-
