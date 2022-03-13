# Tutorial for CPP/CUDA extension of pytorch


## 代码结构
```shell
├── src
│   ├── trillnear_devox_diff_R.cpp # torch wrapper of cuda kernel
│   ├── trillnear_devox_diff_R.h # declaration of cuda kernel(both functions of .cu and .cpp)
│   ├── trillnear_devox_diff_R_cuda.cu # cuda kernel
│   ├── tri_api.cpp #  directory of apis
├── setup.py
└── README.md
```


## 文件关系（共需要四个文件）
- .h 声明.cpp&.cu文件中的函数
- .cu 定义cuda核函数以及cuda函数
- .cpp 包装.cu文件中的函数（其中调用了cuda函数）
- api.cpp 包装.cpp中的函数，创建python模块
- setup.py 编译.cpp & .cu文件


## .so文件的生成以及导入:
- .so文件即为编译后的module，在正常使用时import x（去掉.so的后缀名）即可
- .so文件的生成目录： 由CUDAExtension(name='')确定，例如 name='src.tri_op_cuda'，则生成src/tri_op_cuda.so文件
- 此时在src/目录下,import tri_op_cuda即可导入tri_op_cuda
```python
import tri_op_cuda
```


## 编译失败后再次编译:
1. 删除生成的build/文件夹
```shell
rm -rf build/
```
1. 重新运行setup.py
```shell
python setup.py develop
```



## TIPs:
-  .cu和.cpp文件的命名不能相同，否则setuptool编译会报错（JIT编译可以接受同名.cu&.cpp文件）。参考自Pytorch官方文档[CUSTOM C++ AND CUDA EXTENSIONS](https://pytorch.org/tutorials/advanced/cpp_extension.html#), 'As you can see, it is largely boilerplate, checks and forwarding to functions that we’ll define in the CUDA file. We’ll name this file lltm_cuda_kernel.cu (note the .cu extension!). NVCC can reasonably compile C++11, thus we still have ATen and the C++ standard library available to us (but not torch.h). Note that setuptools cannot handle files with the same name but different extensions, so if you use the setup.py method instead of the JIT method, you must give your CUDA file a different name than your C++ file (for the JIT method, lltm.cpp and lltm.cu would work fine).'


## TODO
- [x] 增加.h, .cu, .cpp文件命名以及其中头文件的说明
- [x] 增加setup.py以及setuptools的说明
- [x] 增加文件结构的规范写法，以及编译生成so文件的正确目录
- [ ] 增加通过@@staticmethod装饰器生成自定义的func & class
- [ ] 增加cuda编程的详细说明（多线程）
- [ ] torch::Tensor现在可以在.cu中以Tensor形式访问，参考Pytorch官方文档[CUSTOM C++ AND CUDA EXTENSIONS](https://pytorch.org/tutorials/advanced/cpp_extension.html#). 此前采用的都是连续内存访问，写的时候容易出错，更新写法以及新教程



---
留言：本文档用于学习交流pytorch中cuda/cpp编程的一些经验，作者本人也是初学者，有错误请及时指出，万分感谢！

@author: Wangjie

Email: jwang991020@gmail.com