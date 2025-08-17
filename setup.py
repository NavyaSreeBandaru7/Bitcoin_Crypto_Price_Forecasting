from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
def get_version():
    version_file = Path(__file__).parent / "crypto_forecasting" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
            if version_match:
                return version_match.group(1)
    return "2.1.0"

# Read long description from README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def get_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="crypto-forecasting-system",
    version=get_version(),
    author="Senior ML Engineer",
    author_email="developer@cryptoforecasting.com",
    description="Advanced Cryptocurrency Price Forecasting System with Deep Learning and AI",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-forecasting-system",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/crypto-forecasting-system/issues",
        "Documentation": "https://github.com/yourusername/crypto-forecasting-system/wiki",
        "Source Code": "https://github.com/yourusername/crypto-forecasting-system",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "gpu": [
            "torch[cuda]>=1.12.0",
            "nvidia-ml-py3>=7.352.0",
        ],
        "api": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "redis>=4.3.0",
            "celery>=5.2.0",
        ],
        "viz": [
            "plotly>=5.10.0",
            "dash>=2.6.0",
            "bokeh>=2.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-forecast=main:main",
            "crypto-config=config:main",
            "crypto-test=tests.test_main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "crypto_forecasting": [
            "data/*.csv",
            "models/*.pth",
            "config/*.json",
            "templates/*.html",
        ],
    },
    zip_safe=False,
    keywords=[
        "cryptocurrency",
        "bitcoin",
        "ethereum",
        "machine learning",
        "deep learning",
        "lstm",
        "transformer",
        "price prediction",
        "technical analysis",
        "sentiment analysis",
        "trading",
        "finance",
        "pytorch",
        "ai",
        "forecasting",
        "time series",
        "ensemble learning",
        "neural networks",
    ],
)
