# ---

title: Optimise CPU by enabling Intel's Math Kernel Library on TensorFlow

authors:

- Erika B. Aranas

tags:

- Intel MKL
- CPU optimisation
- deep learning

created_at: 2018-10-05

updated_at: 2018-10-05

tldr:
- Compiling TensorFlow (TF) with Intel's Math Kernel Library (MKL) support makes **Resnet-50 run 7x faster on CPUs**.
- Cost: 20 minutes, 16 core CPU with at least 32GB memory, and **minimal code change**.
- Resulting wheel can be copied, so only need to build once.

path: Machine Learning/Template

---

If you've ever ran a TensorFlow code on a CPU, chances are you got a warning message like this:
`tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA`

The reason is that TF's default version does not take advantage of the underlying hardware optimisation (e.g. AVX instructions) that Intel has built into their CPU architecture. To fix this, run the following bash script, and 
don't forget to put the latest TF version!

### Optimise your CPU
```python
#!/bin/bash

set -eu

echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update -q
sudo apt-get install -qy openjdk-8-jdk bazel

git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout v1.10.0

env TEST_TMPDIR=/tmp/bazel TF_NEED_JEMALLOC=1 TF_NEED_GCP=0 TF_NEED_HDFS=0 \
    TF_NEED_S3=0 TF_ENABLE_XLA=0 TF_NEED_GDR=0 TF_NEED_VERBS=0 \
    TF_NEED_OPENCL=0 TF_NEED_CUDA=0 TF_NEED_MPI=0 \
    PYTHON_BIN_PATH=/opt/anaconda/envs/Python3/bin/python \
    PYTHON_LIB_PATH=/opt/anaconda/envs/Python3/lib/python3.6/site-packages \
    CC_OPT_FLAGS="-march=native" ./configure
env TEST_TMPDIR=/tmp/bazel bazel build --config=opt --config=mkl \
    //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package ../
cd ../
```
The result of this script is a wheel that can then be installed everytime you spin up a new instance. It should take only around a minute.

```python
pip install msgpack
pip install tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl
```

### Possible issues
- Don't use any less than 32 GB memory: initial attempt with 4 cores, 16GB failed after two hours of waiting.
- Check your instance: if you used a dedicated instance to run the script, installing the resulting wheel on shared instances will give the following error:

```tensorflow/core/platform/cpu_feature_guard.cc:37] The TensorFlow library was compiled to use AVX512F instructions, but these aren't available on your machine.```

This is because dedicated instances are EC2 M5 machines, which means they support AVX-512 instructions, while shared instances are EC2 M4 machines which only support AVX-2.

### Minimal code change
For maximum CPU performance, read TF's documentation: https://www.tensorflow.org/performance/performance_guide#optimizing_for_cpu
