import setuptools
import misso

""" Setup for CPU Version """
with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="",
    version=misso.__version__,
    author="Anand K Subramanian",
    description="",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
            "Programming Language :: Pythong :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",

)
#===========================================================================#

""" Setup for GPU Version """
