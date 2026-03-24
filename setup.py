#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="the-benchmarker",
    version="1.0.0",
    author="Eihabhalaio",
    description="Measure and compare LLM inference performance on your hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eihabhalaio/The-Benchmarker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "rich>=13.0.0",
        "requests>=2.28.0",
        "psutil>=5.9.0",
    ],
    entry_points={
        "console_scripts": [
            "the-benchmarker=benchmark:main",
        ],
    },
)
