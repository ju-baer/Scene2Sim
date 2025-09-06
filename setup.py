from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="scene2sim",
    version="0.1.0",
    author="Scene2Sim Team",
    description="Advanced scene analysis and simulation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/Scene2Sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "web": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "aiofiles>=0.7.0",
        ],
        "viz": [
            "pygame>=2.0.0",
            "matplotlib>=3.3.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "scene2sim=scene2sim.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "scene2sim": [
            "web/templates/*.html",
            "web/static/*.css",
            "web/static/*.js",
        ],
    },
)
