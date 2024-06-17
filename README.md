# 基于numpy的Romberg求积算法实现

本文件包含一个简易的基于numpy的Romberg求积算法实现。关于Romberg算法，请参阅[维基百科](https://en.wikipedia.org/wiki/Romberg%27s_method)。

## Requirements

```
numpy
matplotlib
```

## Code Structure

本项目仅包含一个`main.py`文件，包含全部所需的计算与展示代码。

直接运行`main.py`会计算积分$\int_0^{\frac\pi2}(x^2 + x + 1)\cos(x)$，展示各序列的收敛速度，将各序列及其误差以markdown表格的形式记录在`results.md`文件中，并测试该函数的平均运行时间。


如果想对其他函数进行求积，请确保您的函数支持numpy的向量化操作，或使用`numpy.vectorize`等方法进行包装。具体请参见[numpy文档](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html)