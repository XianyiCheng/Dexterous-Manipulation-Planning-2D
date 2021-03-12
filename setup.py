from setuptools import setup, find_packages

setup(name='kinorrt',
      version='0.1',
      description='Kinodynamic RRT for planning through contacts',
      url='https://github.com/XianyiCheng/kinorrt',
      author='Xianyi CHeng',
      author_email='xianyic@andrew.cmu.edu',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      scripts=[],
      install_requires=[],
      zip_safe=False)