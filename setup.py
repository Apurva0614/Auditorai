from setuptools import setup, find_packages

setup(
    name="auditor-model",
    version="0.1.0",
    description=(
        "Auditor model that suppresses unreliable AI predictions "
        "in human-AI decision systems"
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
)
