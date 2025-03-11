"""
Setup script for fast-dedupe.

This script is used to install the fast-dedupe package.
"""

import os
from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read the version from the package
with open(os.path.join("fastdedupe", "__init__.py"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="fast-dedupe",
    version=version,
    description="Fast, Minimalist Text Deduplication Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/username/fast-dedupe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="deduplication, fuzzy matching, text processing, data cleaning",
    python_requires=">=3.7",
    install_requires=[
        "rapidfuzz>=2.0.0,<3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "fastdedupe=fastdedupe.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/username/fast-dedupe/issues",
        "Source": "https://github.com/username/fast-dedupe",
        "Documentation": "https://github.com/username/fast-dedupe",
    },
) 