from __future__ import print_function
from setuptools.command.install_scripts import install_scripts
import sys, os, tarfile
if sys.version_info < (3, ):
    print("Please use Python 3 for nnf python library")
    sys.exit(-1)

import platform
python_min_version = (3, 6, 2)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print("You are using Python {}. Python >={} is required.".format(
        platform.python_version(), python_min_version_str))
    sys.exit(-1)

nnf_bin="build/src/tools/nnfusion/nnfusion"
nnf_tool="build/src/tools/nnfusion/"
nnf_resource=os.listdir(nnf_tool)
nnf_blacklist = ["cmake_install.cmake", "Makefile", "CMakeFiles"]
nnf_pkg = "src/python/nnfusion/nnfusion.tar.gz"

if not os.path.exists(nnf_bin):
    print("Pleast build nnfusion in /build folder.")
    sys.exit(-1)

def make_targz(output_filename, source_dir, ignore_list): 
    tar = tarfile.open(output_filename,"w:gz")
    for root, dir, files in os.walk(source_dir):
        skip = False
        for word in ignore_list:
            if word in root:
                skip = True
                break
        if skip:
            continue
        
        for file in files:
            if file in ignore_list:
                continue
            pathfile = os.path.join(root, file)
            tar.add(pathfile, arcname=pathfile.replace("build/src/tools/nnfusion/", ""))
    tar.close()

make_targz(nnf_pkg, nnf_tool, nnf_blacklist)

import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("requirements_test.txt") as f:
    tests_require = f.read().splitlines()

with open("VERSION") as f:
    version = f.readline().strip()

print(setuptools.find_packages(where="src/python", exclude=["example"]))

setuptools.setup(
    name="nnfusion",
    version=version,
    author="Microsoft Corporation",
    author_email="nnfusion-team@microsoft.com",
    description="NNFusion is a flexible and efficient DNN compiler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/nnfusion",
    packages=setuptools.find_packages(where="src/python", exclude=["example"]),
    package_dir={"": "src/python"},
    package_data={"":["nnfusion.tar.gz"]},
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
    ]
    )
