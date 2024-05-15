# 1 需要修改的地方

- dataset\transform.py

  normalize函数采用了ImageNet上预训练模型常见的预定义平均值和标准差值对图像进行标准化。不确定是否需要修改，需要确认一下；
- 如果要适应多通道tif数据，需要对semi和transform两个py程序进行彻底修改

# 2 不懂的地方

unimatch.py程序中训练循环很复杂，需要重新分析