from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("pcf", ["pcfmodule.cpp","../src/LatentFactorModel.cpp"],
	                        include_dirs = ['/usr/local/include/eigen3'])

setup(name = "pcf", ext_modules=[extension_mod])