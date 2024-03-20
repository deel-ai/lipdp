# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -*- encoding: utf-8 -*-
import os

from setuptools import find_namespace_packages
from setuptools import find_packages
from setuptools import setup

REQ_PATH = "requirements.txt"
REQ_DEV_PATH = "requirements_dev.txt"

install_requires = []

if os.path.exists(REQ_PATH):
    print("Loading requirements")
    with open(REQ_PATH, encoding="utf-8") as fp:
        install_requires = [line.strip() for line in fp]

dev_requires = [
    "setuptools",
    "pre-commit",
    "pytest",
    "tox",
    "black",
    "pytest",
    "pylint",
    "mkdocs",
    "mkdocs-material",
    "mkdocstrings[python]",
    "mknotebooks",
    "bump2version",
]

README_PATH = "README.md"
readme_contents = ""
if os.path.exists(README_PATH):
    with open(README_PATH, encoding="utf8") as fp:
        readme_contents = fp.read().strip()

with open("deel/lipdp/VERSION", encoding="utf8") as f:
    version = f.read().strip()

setup(
    # Name of the package:
    name="lipdp",
    # Version of the package:
    version=version,
    # Find the package automatically (include everything):
    packages=find_namespace_packages(include=["deel.*"]),
    package_data={"": ["VERSION"]},
    include_package_data=True,
    # Author information:
    # Author information:
    author="",
    author_email="",
    # Description of the package:
    description="Differentially Private 1-Lipschitz network building",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    # Plugins entry point
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    licence="MIT",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
    },
)
