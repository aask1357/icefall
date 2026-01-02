from setuptools import setup
from torch.utils import cpp_extension


def no_op_check(compiler_name, compiler_version):
    # 아무것도 하지 않고 통과시킴
    pass

# PyTorch의 버전 체크 함수를 우리의 '빈 함수'로 덮어씌웁니다.
cpp_extension._check_cuda_version = no_op_check

setup(
    name=f'omniquant_cuda',
    version='0.1.0',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'kernels.omniquant_cuda',
            ['kernels/omniquant.cpp', 'kernels/omniquant_cuda.cu'],
            extra_compile_args={
                'cxx': ['-g', '-march=native', '-funroll-loops'],
                'nvcc': ['-O3', '-lineinfo', '--fmad=true', '-U__CUDA_NO_HALF_CONVERSIONS__',
                         '-U__CUDA_NO_HALF_OPERATORS__']
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

# USAGE:
# $ python setup_iir_2o.py build_ext --inplace

# If error occurs (ex. cannot find -lcudart: No such file or directory)
# try downlaoding & installing cuda toolkit from https://developer.nvidia.com/cuda-toolkit-archive
# and add the path to the cuda toolkit to the environment variable PATH
# ex. if you downloaded & installed cuda toolkit 11.7, type below to the terminal:
# $ export PATH=/usr/local/cuda-11.7/bin:$PATH