from setuptools import setup


setup(name='convolution-kernels-numpy',
      version='0.1',
      description='Python package that implements kernels/filters convolutions for time series modeling.',
      url='https://github.com/diegoarri91/convolution-kernels-numpy',
      author='Diego M. Arribas',
      author_email='diegoarri91@gmail.com',
      license='MIT',
      packages=['kernel'],
      install_requires=['matplotlib', 'numpy', 'scipy']
      )
