# Information Accuracy

This repo contains some modifications and additions to the original code of the [simple MNIST tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py).
 
The modification is described in the paper "Entropy Non-increasing Games for Improvement of Dataflow Programming"
This paper is a part of the [ESAMU](https://github.com/nbatfai/SamuEntropy) research program.

## Some Running Results

batch size = 1000

```
nbatfai@robopsy:~/Robopsychology/forkedrepos/tensorflow/tensorflow/examples/tutorials/mnist$ python mnist_softmax_esmu.py 
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so.8.0 locally
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:909] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, s
o returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GTX 660 Ti
major: 3 minor: 0 memoryClockRate (GHz) 1.0455
pciBusID 0000:01:00.0
Total memory: 1.95GiB
Free memory: 1.28GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 660 Ti, pci bus id: 0000:01:00.0)
-- Train
0.0 %
10.0 %
20.0 %
30.0 %
40.0 %
50.0 %
60.0 %
70.0 %
80.0 %
90.0 %
--     Acc:  9.29000020027e-06
-- InfoAcc:  2.62447668517  bit
----------------------------------------------------------
-- Test
--     Acc:  0.9249
-- InfoAcc:  2.55959  bit
----------------------------------------------------------

```

batch size = 1000

```
nbatfai@robopsy:~/Robopsychology/forkedrepos/tensorflow/tensorflow/examples/tutorials/mnist$ python mnist_softmax_esmu.py 
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so.8.0 locally
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:909] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, s
o returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GTX 660 Ti
major: 3 minor: 0 memoryClockRate (GHz) 1.0455
pciBusID 0000:01:00.0
Total memory: 1.95GiB
Free memory: 1.45GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 660 Ti, pci bus id: 0000:01:00.0)
-- Train
0.0 %
10.0 %
20.0 %
30.0 %
40.0 %
50.0 %
60.0 %
70.0 %
80.0 %
90.0 %
--     Acc:  9.37600135803e-06
-- InfoAcc:  2.62743278917  bit
----------------------------------------------------------
-- Test
--     Acc:  0.9249
-- InfoAcc:  2.5599  bit
----------------------------------------------------------

```

batch size = 100

```
nbatfai@robopsy:~/Robopsychology/forkedrepos/tensorflow/tensorflow/examples/tutorials/mnist$ python mnist_softmax_esmu.py 
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so.8.0 locally
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:909] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GTX 660 Ti
major: 3 minor: 0 memoryClockRate (GHz) 1.0455
pciBusID 0000:01:00.0
Total memory: 1.95GiB
Free memory: 1.24GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 660 Ti, pci bus id: 0000:01:00.0)
-- Train
0.0 %
10.0 %
20.0 %
30.0 %
40.0 %
50.0 %
60.0 %
70.0 %
80.0 %
90.0 %
--     Acc:  9.30000066757e-06
-- InfoAcc:  2.59557406819  bit
----------------------------------------------------------
-- Test
--     Acc:  0.9244
-- InfoAcc:  2.55578  bit
----------------------------------------------------------
```


