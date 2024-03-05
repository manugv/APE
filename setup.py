from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


extensions = [Extension(name="functions", sources=["APE/functions.pyx"],
                        include_dirs=[numpy.get_include()])]
setup(
    name="functions",
    ext_modules=cythonize(extensions),
)
