from setuptools import find_packages, setup

setup(
    name="ex_matgl",
    version="0.0.1",
    author="Masaya Hagai",
    license="MIT",
    description="PyTorch implementation for excitation energy and force",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "dgl>=1.0.1",
        "matgl>=0.8.0",
    ],
)