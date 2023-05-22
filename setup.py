# -*- coding: utf-8 -*-
# Copyright anonymized et anonymized - All
# rights reserved. anonymized is a research program operated by anonymized, anonymized,
# anonymized and anonymized - https://www.anonymized.ai/
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
from setuptools import setup

this_directory = os.path.dirname(__file__)
req_path = os.path.join(this_directory, "requirements.txt")
req_dev_path = os.path.join(this_directory, "requirements_dev.txt")

install_requires = []

if os.path.exists(req_path):
    with open(req_path) as fp:
        install_requires = [line.strip() for line in fp]

if os.path.exists(req_dev_path):
    with open(req_dev_path) as fp:
        install_dev_requires = [line.strip() for line in fp]

readme_path = os.path.join(this_directory, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf8") as fp:
        readme_contents = fp.read().strip()

with open(os.path.join(this_directory, "lipdp/VERSION"), encoding="utf8") as f:
    version = f.read().strip()

setup(
    # Name of the package:
    name="lipdp",
    # Version of the package:
    version=version,
    # Find the package automatically (include everything):
    packages=find_namespace_packages(include=["anonymized.*"]),
    package_data={},
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    licence="MIT",
    install_requires=install_requires,
    extras_require={
        "dev": install_dev_requires,
    },
)
