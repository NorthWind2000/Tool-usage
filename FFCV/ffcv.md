# 一、环境安装

## 1.1 Linux

```bash
conda create -y -n ffcv python=3.9
conda activate ffcv
pip install xxxx pytorch
conda install pkg-config libjpeg-turbo numba
conda install opencv
pip install ffcv
```

官方文档：[Quickstart — FFCV documentation](https://docs.ffcv.io/quickstart.html)

> python 版本需要3.9及以上。
>
> pkg-config,libjpeg-turbo,numba,opencv 需要使用conda安装，pip找不到。可以一个一个的安装。这几个包安装起来比较慢，多试几次，中间有可能出现网络不太好的情况。
>
> 对于`opencv`可以尝试从 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge处下载。

推荐豆瓣镜像：https://pypi.douban.com/simple/

## 1.2 Windows

- Install [opencv4](https://opencv.org/releases/)

  - 添加 `..../opencv/build/x64/vc15/bin` 到系统环境变量的路径下

- Install [libjpeg-turbo](https://sourceforge.net/projects/libjpeg-turbo/files/) 下载 `libjpeg-turbo-x.x.x-vc64.exe`, 不带 `gcc64`

  - 添加 `..../libjpeg-turbo64/bin` 到系统环境变量的路径下

- Install [pthread](https://www.sourceware.org/pthreads-win32/) 下载 最新 release.zip  镜像网址：http://sourceware.org/mirrors.html，下面是清华镜像站下载的。

  ![image.png](https://s2.loli.net/2023/06/08/ReDqgumObnM85vX.png)

  - 之后解压, 将里面的`Pre-build.2`文件夹重命名为`pthread`
  - 打开 `pthread/include/pthread.h`, 并添加下面代码在文件的顶部。

  ```c
  #define HAVE_STRUCT_TIMESPEC
  ```

  - 添加 `..../pthread/dll/x64` 到系统环境变量的路径下

> 注意：安装的时候，会寻找`pthread`的位置，有时候可能以为匹配到其他路径而报 找不到。

- Install [cupy](https://docs.cupy.dev/en/stable/install.html#installing-cupy) depending on your CUDA Toolkit version.

- `pip install ffcv`

**设置环境变量后，重启一下。**

如果提示下面错误：

![image.png](https://s2.loli.net/2023/06/08/3YL7BpPk6s2yTgJ.png)

进入：[Microsoft C++ 生成工具 - Visual Studio](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)

安装C++：

![image.png](https://s2.loli.net/2023/06/08/tbDun57owFf64Ns.png)

**安装过于麻烦，使用体验较差！**
**FFCV的加速主要是针对数据加载和数据增广。**

# 二、将数据集转为FFCV格式

## 2.1 支持数据集格式

首先，需要将我们自己的数据集格式（pytorch格式）转为FFCV格式。

FFCV支持将下面两种格式转为自身格式：

1. 可索引对象：它们需要实现一个函数，将与样本相关的数据作为元组/列表(任意长度)返回。
2. Webdataset：网络数据集，允许用户更容易的使用网络上大规模的数据。

本教程讲解第一种格式`可索引对象`。

第一步，在脚本中加入下面代码：

```python
from ffcv.writer import DatasetWriter
```

## 2.2 Indexable Dataset

### 2.2.1 构建可索引数据集

对于这个例子，我们将构建一个简单的线性回归数据集，它返回一个输入向量和它对应的标签:

```python
import numpy as np

class LinearRegressionDataset:
    def __init__(self, N, d):
        self.X = np.random.randn(N, d)
        self.Y = np.random.randn(N)

    def __getitem__(self, idx):
        return (self.X[idx].astype('float32'), self.Y[idx])

    def __len__(self):
        return len(self.X)

N, d = (100, 6)
dataset = LinearRegressionDataset(N, d)
```

> 这里我们最常用的就是pytorch自带的`torch.utils.data.Dataset`

### 2.2.2 构建写入器

负责将其他数据集转为FFCV格式的是：`ffcv.writer.DatasetWriter`

该方法需要的参数：

- 一个路径：`.beton`写入的位置；（beton是ffcv数据集的文件后缀）
- 一个字典：将key映射到fields。（field=字段）

每个字段对应于数据集返回的数据元组中的一个元素，并指定该元素应该如何写入FFCV数据集文件。在我们的示例中，数据集有两个字段，**一个用于(向量)输入，另一个是相应的(标量)标签**。这两个字段已经在FFCV中有默认实现，我们在下面使用：

```python
from ffcv.fields import NDArrayField, FloatField

writer = DatasetWriter(write_path, {
    'covariate': NDArrayField(shape=(d,), dtype=np.dtype('float32')),
    'label': FloatField(),
}, num_workers=16)
```

> 注意：转换时，保证这里标注的`dtype`和数据真实类型一致。

### 2.2.3 写入FFCV数据集

构造好写入器后，便可以执行下面代码写入数据集：

```python
writer.from_indexed_dataset(my_dataset)
```

### 2.2.4 Fields 字段

除了上面使用的例子，FFCV支持各种内置的字段，这使得直接转换大多数数据集变得很容易:

- `RGBImageField`：处理图像，包括(可选的)压缩和调整大小。传入一个PyTorch张量。
- `IntField` 和 `FloatField`：处理简单的标量字段。传入int或float。
- `BytesField`：存储可变长度的字节数组。传入numpy字节数组。
- `JSONField`：编码JSON文档，传入能够JSON编码的字典。

# 三、制作FFCV数据加载器（dataloader）

为了加载我们已经写入的数据集，我们需要使用`ffcv.loader.Loader`类和一组与数据集字段相对应的解码器。例如：`FloatDecoder`和`NDArrayDecoder`。

```python
from ffcv.loader import Loader,OrderOption
from ffcv.fields.decoders import NDArrayDecoder,FloatDecoder
```

首先，第一步实例化`Loader`类。

```python
loader = Loader('dataset.beton',
               batch_size=BATCH_SIZE,
               num_workers=NUM_WORKERS,
               order=ORDERING,
               pipelines=PIPELINES)
```

## 3.1 Dataset ordering

加载器初始化中的`order`选项，类似于PyTorch DataLoader的`shuffle`选项，但有一些额外的选项。该参数接受`ffcv.loader.OrderOption`提供的enum:

```python
from ffcv.loader import OrderOption

ORDERING = OrderOption.RANDOM
ORDERING = OrderOption.SEQUENTIAL
ORDERING = OrderOption.QUASI_RANDOM
```

> RANDOM：最需要内存，因为它必须缓存整个数据集以随机采样。如果可用内存不够，它将抛出异常。
>
> QUASI_RANDOM：提前缓存一部分样本。需要的内存比RANDOM少得多，但比SEQUENTIAL多一点。当整个数据集无法容纳RAM时使用它。
>
> SEQUENTIAL：需要最少的内存。它只在输入训练迭代之前加载几个样本。

## 3.2 Pipelines

Loader中的`pipeline`选项指定数据集，并告诉加载器要读取哪些字段、如何读取它们以及对该字段应用哪些处理操作。

具体来说，管道是一个`键-值`字典，其中键与写入数据集时使用的键匹配，值是要执行的操作序列。操作必须以该字段对应的`解码器对象`开头，后面跟一系列转换。

例如，下面的管道读取字段，然后将每个字段转换为PyTorch张量：

```python
from ffcv.transforms import ToTensor

PIPELINES = {
  'covariate': [NDArrayDecoder(), ToTensor()],
  'label': [FloatDecoder(), ToTensor()]
}
```

## 3.3 Tranforms

在管道中利用数据变换有三种简单的方式：

1. `ffcv.transforms`中的一组标准变换。这里包括图像增强，例如：`RandomHorizontalFlip`。
2. 任何`torch.nn.Module`的子类，FFCV会自动将其转为一个操作。
3. 自定义变换操作：可以通过`ffcv.transforms.Operation`。



