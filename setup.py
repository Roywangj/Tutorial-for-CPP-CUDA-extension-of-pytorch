from setuptools import setup
import os
# import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
# import subprocess

# def get_git_commit_number():
#     if not os.path.exists('.git'):
#         return '0000000'

#     cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
#     git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
#     return git_commit_number



if __name__ == '__main__':
    include_dirs = os.path.dirname(os.path.abspath(__file__))
    # include_dirs = os.path.join(include_dirs, "innnn")

    # version = '0.1.0+%s' % get_git_commit_number() 
    # source_cpu = glob.glob(os.path.join(include_dirs, 'gpu', '*.cpp', "*.cu"))
    setup(
        name='tril_devox_gpu',
        version="0.1",
        author='Jie Wang',
        author_email='jwang991020@gmail.com',
        ext_modules=[
            CUDAExtension(
            name = 'src.tri_op_cuda', 
            sources=["src/tril_api.cpp",
                    "src/trillnear_devox_diff_R.cpp",
                    "src/trillnear_devox_diff_R_cuda.cu"], 
            include_dirs=[include_dirs]),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )


# from setuptools import setup
# import os
# # import glob
# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
# # import subprocess

# # def get_git_commit_number():
# #     if not os.path.exists('.git'):
# #         return '0000000'

# #     cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
# #     git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
# #     return git_commit_number



# if __name__ == '__main__':
#     include_dirs = os.path.dirname(os.path.abspath(__file__))
#     # version = '0.1.0+%s' % get_git_commit_number() 
#     # source_cpu = glob.glob(os.path.join(include_dirs, 'gpu', '*.cpp', "*.cu"))
#     setup(
#         name='tril_devox_gpu',
#         version="0.1",
#         ext_modules=[
#             CUDAExtension('tril_devox_gpu', sources=["src/trillnear_devox_diff_R.cpp", "src/trillnear_devox_diff_R_cuda.cu"], include_dirs=[include_dirs]),
#         ],
#         cmdclass={
#             'build_ext': BuildExtension
#         }
#     )