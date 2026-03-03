"""
setup.py — minimal package configuration for agi-mvp-arc-agi-1.

Install in development mode:
    pip install -e .

This makes the package importable from anywhere without path hacking.
"""
from setuptools import setup, find_packages

setup(
    name="agi-mvp-arc-agi-1",
    version="0.1.0",
    description="MDL-guided symbolic search for ARC-AGI and beyond",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "scripts*", "docs*"]),
    install_requires=[],   # zero dependencies — pure stdlib
    extras_require={
        "dev": ["pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
