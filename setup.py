from setuptools import setup, find_packages
from pathlib import Path

__doc__ = """
Bayesian Consistency Networks for Contradiction Resolution

A Python implementation of Bayesian Consistency Networks for resolving contradictions
in information from multiple unreliable sources using probabilistic graphical models.
"""

# Read the contents of README.md
here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read version from package
version = {}
with open(here / "bcn" / "__init__.py", encoding="utf-8") as f:
    exec(f.read(), version)

setup(
    name="bayesian-consistency-networks",
    version=version.get("__version__", "0.1.0"),
    author="LexSight LLC",
    author_email="info@lexsight.ai",
    description="Bayesian Consistency Networks for Contradiction Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lexsightllc/bayesian-consistency-networks",
    project_urls={
        "Bug Tracker": "https://github.com/lexsightllc/bayesian-consistency-networks/issues",
        "Documentation": "https://github.com/lexsightllc/bayesian-consistency-networks#readme",
        "Source Code": "https://github.com/lexsightllc/bayesian-consistency-networks",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "bcn": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
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
            'pylint>=2.12.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'myst-parser>=0.15.0',
        ],
    },
    keywords=[
        'bayesian-inference',
        'consistency',
        'contradiction-detection',
        'belief-propagation',
        'probabilistic-graphical-models',
        'machine-learning',
    ],
    license='Apache 2.0',
    license_files=('LICENSE',),
)
