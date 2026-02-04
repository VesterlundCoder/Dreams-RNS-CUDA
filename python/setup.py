from setuptools import setup, find_packages

setup(
    name="dreams-rns",
    version="0.1.0",
    description="GPU-accelerated Ramanujan Dreams pipeline using RNS arithmetic",
    author="VesterlundCoder",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "sympy>=1.9",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "analysis": ["matplotlib>=3.4.0", "pandas>=1.3.0"],
        "precision": ["mpmath>=1.2.0"],
        "dev": ["pytest>=6.0.0"],
    },
      extra_cuda_cflags=[
        "-03",
        "--use_fast_math",
        "--maxrregcount=80",
       "--lineinfo",
    ]
)
