from setuptools import setup
from torch.utils import cpp_extension


setup(
    name=f'iir_cuda',
    version='1.0.0',
    ext_modules=[
        cpp_extension.CUDAExtension(
            f'kernels.iir_cuda',
            [f'kernels/iir.cpp', f'kernels/iir_cuda.cu'],
            extra_compile_args={
                'cxx': ['-g', '-march=native', '-funroll-loops'],
                'nvcc': ['-O3', '-lineinfo', '--fmad=true']
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

# USAGE:
# $ python setup_iir.py build_ext --inplace

# If error occurs (ex. cannot find -lcudart: No such file or directory)
# try downlaoding & installing cuda toolkit from https://developer.nvidia.com/cuda-toolkit-archive
# and add the path to the cuda toolkit to the environment variable PATH
# ex. if you downloaded & installed cuda toolkit 11.7, type below to the terminal:
# $ export PATH=/usr/local/cuda-11.7/bin:$PATH