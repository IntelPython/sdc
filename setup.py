from setuptools import setup, Extension

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='hpat',
      version='0.1.0',
      description='compiling Python code for clusters',
      long_description=readme(),
      classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
      ],
      keywords='data analytics cluster',
      url='https://github.com/IntelLabs/hpat',
      author='Ehsan Totoni',
      author_email='ehsan.totoni@intel.com',
      packages=['hpat'],
      install_requires=[
          'numba',
      ])
