from setuptools import setup, find_packages, Extension


spkmeans = Extension('spkmeans', sources=['spkmeansmodule.c'])

setup(name = "spkmeans",
    version="1.0",
    author="Nitzan and Nir",
    author_email="nitzanyizhar@mail.tau.ac.il",
    description="Ultra mega +++++SPKMeans+++++",
    install_requires=['inovke'],
    packages=find_packages(),
    ext_modules=[spkmeans]
)