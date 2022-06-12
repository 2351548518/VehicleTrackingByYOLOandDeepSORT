[(1条消息) YOLOv5算法详解_技术挖掘者的博客-CSDN博客_yolov5算法](https://blog.csdn.net/WZZ18191171661/article/details/113789486)

YOLOv1

标签中心落在那个grid cell中，就由那个grid cell负责预测

![image-20220604145112001](https://raw.githubusercontent.com/2351548518/images/main/CCSE/202206041451173.png)

每个grid cell只能预测一个物体/类别

对小物体预测不好

每个grid cell 生成两个 badding box预测

然后进行减少预测框IOU和概率

