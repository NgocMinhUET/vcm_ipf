from setuptools import setup, find_packages

setup(
    name="ipf-phase2",
    version="0.1.0",
    description="IPF Phase 2: VTM/VVC Integration and Encoding Experiments",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "matplotlib>=3.5",
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "opencv-python-headless>=4.5",
        "ultralytics>=8.0",
        "tqdm>=4.60",
        # Phase 3 Stage C — LiteQP residual MLP trainer + model persistence.
        "scikit-learn>=1.0",
        "joblib>=1.1",
    ],
)
