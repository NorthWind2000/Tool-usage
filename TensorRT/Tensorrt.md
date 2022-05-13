**Ubnutu版本安装：**

Ubnutu18.04

Cuda 11.1

python 3.7.1

pytorch 1.10.1

Tensorrt版本 8.2.2.1

他们之间的版本关系：只要Cuda和pytorch相互对应，Tensorrt和Cuda相互对应即可。

查看Cuda和pyorch对应关系：https://pytorch.org/get-started/previous-versions/

查看Tensorrt和Cuda对应关系：https://developer.nvidia.com/nvidia-tensorrt-download

cuDNN 11.1

如果下载的是deb的Tensorrt安装包需要CUDA也是用deb安装的，否则安装会报错。 

deb版本CUDA安装：

sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

deb Tensorrt安装方式：

sudo dpkg -i nv-tensorrt-repo-ubuntu1x04-cudax.x-trt5.x.x.x-ga-yyyymmdd_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-cudax.x-trt5.x.x.x-ga-yyyymmdd/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt

sudo apt-get install python3-libnvinfer-dev

以上步骤完成后，无法在python中import Tensorrt，需要去官网下一个tar版的Tensorrt的压缩包，将其解压后，进入其python文件夹中，pip install 对应python的whl文件



WIN10系统

电脑配置：pytorch：1.9.0 CUDA：10.2  cuDANN:8.1.0

下载的Tensorrt版本：8.2.4.2

1.解压下载的压缩包

2.将解压后的文件夹中的lib文件夹加入到环境变量中

3.

```
cd TensorRT-${version}/python  #cd到解压后的文件夹中的python文件夹中
python3 -m pip install tensorrt-8.2.4.2-cp38-none-win_amd64.whl#根据自己的python版本选择安装的文件
```

4.

```
cd TensorRT-${version}/graphsurgeon
python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl #版本号
```

5.

```
cd TensorRT-${version}/onnx_graphsurgeon
	
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
```

6.安装完成后通过输入

```
import tensorrt as trt

trt.__version__
```

检查是否安装成功

7.pip install cuda-python

pip install opencv-python

pip install onnxruntime==1.10.0

8.将pt模型转换为onnx，并进行检验

```python
net = model()
net.eval()
path = "./net.pt"  
onnxFile='./model.onnx'
checkpoint = torch.load(path,map_location='cpu')  # 加载模型
net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
input=torch.randn(1,3,256,256)#构造的输入
output=net(input)

torch.onnx.export(net,input,'./model.onnx',input_names=['input'],output_names=['output'],verbose=True,opset_version=9)#把.pt模型转换成.onnx
ort_session=onnxruntime.InferenceSession('./model.onnx')
ort_inputs={ort_session.get_inputs()[0].name:input.detach().numpy()}
ort_outs=ort_session.run(None,ort_inputs)
#np.testing.assert_allclose(output.detach().numpy(),ort_outs,rtol=1e-3,atol=1e-5) 检验onnx模型和pt模型的输出是否相同
```

9.构建Tensorrt,创建logger

```python
logger = trt.Logger(trt.Logger.ERROR)#Logger
if os.path.isfile(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)#Builder
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))#使用Explicit Batch模式
    profile = builder.create_optimization_profile()#帮助网络优化
    config = builder.create_builder_config()#builder config 设置网络属性
    config.max_workspace_size = 3 << 30#指定构建期可用显存（单位Byte)
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
```

10.并初始化引擎，设置输入，及其shape

```python

inputTensor = network.get_input(0)#设置一个输入张量
    profile.set_shape(inputTensor.name, (4, 3, 256, 256), (4, 3, 256, 256), (4, 3, 256, 256))#张量最小 最常见 最大的尺寸
    config.add_optimization_profile(profile)
    engineString = builder.build_serialized_network(network, config)#生成TRT内部表示
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)#生成Engine

context = engine.create_execution_context()#创建Context 类比进程
context.set_binding_shape(0, [4, 3, 256, 256])#设置张量真正运行的形状 只需要设置输入的binding 输出的binding形状会自动生成
_, stream = cudart.cudaStreamCreate()
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))
data=input
#data = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE).astype(np.float32)
#准备Buffer
inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
```

11.执行计算，最后释放内存

```python
# 执行计算
cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("inputH0 :", data.shape)
#print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
print("Succeeded running model in TensorRT!")
```

仍存在的问题：

 [executionContext.cpp::nvinfer1::rt::ExecutionContext::enqueueInternal::330] Error Code 3: API Usage Error (Parameter check failed at: executionContext.cpp::nvinfer1::rt::ExecutionContext::enqueueInternal::330, condition: bindings[x] != nullptr

运行的时候发现显存一瞬间拉满了，可能是显存爆了，换到Ubnutu安装后，成功运行。

