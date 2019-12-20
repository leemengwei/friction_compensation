from distutils.core import setup
from Cython.Build import cythonize
from glob import glob
from IPython import embed

#python 'this' build_ext  --inplace

pys = []
pys += glob("*.py")
#pys += glob("*/*.py")


for py in pys:
    setup(ext_modules=cythonize([py]))

print("Done")
