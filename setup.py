# setup.py

from setuptools import setup, find_packages

setup(
    name="lung_segmentator",
    version="0.1.0",
    description="A lightweight U-ViT-based lung segmentation package for fine-tuning, analysis, and experimentation",
    author="Arav Jha",
    author_email="linrajesh@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.11.0",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tensorflow-datasets"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)