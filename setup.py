from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("function", "src/ape/functions.pyx")]
setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
