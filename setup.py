from setuptools import setup, find_packages
import pathlib

# Read the contents of README.md
here = pathlib.Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="bayesian-consistency-networks",
    version="0.1.0",
    author="Augusto Lex",
    author_email="augusto.lex@example.com",
    description="Bayesian Consistency Networks for Contradiction Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/augustolex/bayesian-consistency-networks",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "bcn": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
            'mypy>=0.910',
            'black>=21.7b0',
            'isort>=5.9.0',
            'flake8>=3.9.0',
        ],
    },
)
