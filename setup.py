# !/usr/bin/env python

from distutils.core import setup, Extension
import setuptools
import os
import platform

"""
code from:
https://pythonextensionpatterns.readthedocs.io/en/latest/compiler_flags.html
"""
import sysconfig
import torch

from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension


BASE_DIR  = os.path.split(os.path.realpath(__file__))[0]
_DEBUG = False
# Generally I write code so that if DEBUG is defined as 0 then all optimisations
# are off and asserts are enabled. Typically run times of these builds are x2 to x10
# release builds.
# If DEBUG > 0 then extra code paths are introduced such as checking the integrity of
# internal data structures. In this case the performance is by no means comparable
# with release builds.
_DEBUG_LEVEL = 0

# Common flags for both release and debug builds.
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra"]
if _DEBUG:
    extra_compile_args += ["-g3", "-O0", "-DDEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
else:
    extra_compile_args += ["-DNDEBUG", "-O3"]

def make_cpp_ext(name:str,module:str,sources:list,include_dirs:list,library_dirs:list=[],**kwargs):
    return CppExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs = [os.path.join(BASE_DIR,*module.split("."),p) for p in include_dirs],
        ##extra_link_args =sysconfig.get_config_vars('LDFLAGS'),
        **kwargs
    )

def make_cuda_ext(name, module, sources=[], sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = Extension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

if __name__ == '__main__':
    setup(
        name='texthub',
        package_data={'texthub.ops': ['*/*.so']},
        ext_modules=[
            make_cpp_ext(
                name="pan_cpp",
                module="texthub.ops.pan",
                sources=["src/pse.cpp"],
                include_dirs=['include/pybind11'],
                extra_compile_args=extra_compile_args,
                language='c++11',

            ),
            make_cuda_ext(
                name='deform_conv_cuda',
                module="texthub.ops.dcn",
                sources=[],
                sources_cuda=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_cuda',
                module="texthub.ops.dcn",
                sources=[],
                sources_cuda=[
                     'src/deform_pool_cuda.cpp',
                    'src/deform_pool_cuda_kernel.cu'
                ])
            ],
        cmdclass={'build_ext': BuildExtension}
    )


