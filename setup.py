"""
To automatically increment the version number and package, run:
    python setup.py sdist version++

To push to pip test:
    twine upload --repository testpypi dist/*
"""
from setuptools import setup, find_packages
import sys

def increment_version(old: str):
    s = [int(i) for i in old.split(".")]
    s[-1] += 1
    return ".".join(str(i) for i in s)

if "version++" in sys.argv:
    sys.argv.remove("version++")
    with open("src/backtest/__version__.py", "r") as f:
        version = f.read().split("=")[1].strip().strip("\"")

    with open("src/backtest/__version__.py", "w") as f:
        f.write(f"__version__ = \"{increment_version(version)}\"")

with open("PYPI.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Load version
with open("src/backtest/__version__.py", "r") as f:
    version = f.read().split("=")[1].strip().strip("\"")

setup(
    name="backtest-pro",
    version=version,
    author="Anthony Lavertu",
    author_email="alavertu2@gmail.com",
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.css', '*.html', "LICENSE"],
    },
    description="A feature-rich event driven backtesting framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthol42/backtestPro",
    project_urls={
        "Issues": "https://github.com/anthol42/backtestPro/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords=[
        "backtest-pro", "backtestpro", "backtest", "pybacktest", "py-backtest", "backtesting", "quant", "finance",
        "stocks", "crypto", "cryptocurrency", "derivatives", "trading", "investing", "financial",
        "technical", "fundamental", "ai", "machine", "learning", "neural", "network", "deep", "reinforcement", "algorithm",
        "strategy", "portfolio", "optimization", "risk", "management"
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.4",
        "matplotlib>=3.8.0",
        "psutil>=5.9.0",
        "py-cpuinfo>=9.0.0",
        "mplfinance>=0.12.10b0",
        "tqdm>=4.66.0",
        "yfinance>=0.2.37"
    ],
    extras_require={
        'optional': [
            "TA-Lib>=0.4.0",
            "plotly>=5.20.0",
            "schedule>=1.2.0",
            "kaleido>=0.2.1",
            "weasyprint>=61.2",
            "python-crontab>=3.0.0"
        ],
    },
    entry_points={
        "console_scripts": [
            "backtest = backtest.main:main",
        ],
    },
    # packages=find_packages(),
)
