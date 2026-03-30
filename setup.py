"""
TEKNOFEST 2026 — Onkolojide 3T
GlioSight: Multimodal AI ile Glioblastoma Tanı ve Tedavi Yanıt Tahmini Platformu
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gliosight",
    version="0.1.0",
    author="GlioSight Team",
    description=(
        "Multimodal AI platform for glioblastoma diagnosis, "
        "segmentation, and treatment response prediction"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bahattinyunus/teknofest_onkolojide_3t",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "monai>=1.3.0",
        "nibabel>=5.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "lifelines>=0.27.0",
        "xgboost>=2.0.0",
        "shap>=0.43.0",
        "pyradiomics>=3.1.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "plotly>=5.17.0",
            "nilearn>=0.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="brain tumor glioblastoma MRI segmentation survival prediction AI oncology",
)
