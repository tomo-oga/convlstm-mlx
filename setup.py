from setuptools import setup, find_packages

setup(
    name='convlstm-mlx',
    version='0.1.1-alpha',
    description='An MLX implementation of ConvLSTM',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Tomohiro Oga',
    author_email='oga.t@northeastern.edu',
    url='https://github.com/tomo-oga/convlstm-mlx',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'mlx',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.7'
)