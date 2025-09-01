from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Scene2Sim",
    version="0.1.0",
    description="Annotation-native lightweight AV simulator for counterfactual evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AV Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pygame>=2.0.0", 
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "full": ["pandas>=1.3.0", "seaborn>=0.11.0"],
        "export": ["lxml>=4.6.0"],  # For OpenSCENARIO
        "dev": ["pytest>=6.0.0", "black>=21.0.0", "flake8>=3.9.0"],
    },
    entry_points={
        "console_scripts": [
            "adsimlite-demo=examples.demo_basic:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
