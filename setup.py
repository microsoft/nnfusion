from __future__ import print_function
import sys
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

import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("requirements_test.txt") as f:
    tests_require = f.read().splitlines()

with open("VERSION") as f:
    version = f.readline().strip()

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
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
    ])
