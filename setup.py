from setuptools import setup, find_packages

setup(
    name="ipf-phase1",
    version="0.1.0",
    description="Tracker-Informed Spatiotemporal Importance Field - Phase 1 Precompute",
    author="IPF Research Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "opencv-python>=4.8.0",
        "ultralytics>=8.1.0",
        "torch>=2.0.0",
        "PyYAML>=6.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "ipf-run=phase1.cli.run:app",
            "ipf-viz=phase1.cli.visualize:app",
            "ipf-compare=phase1.cli.compare:app",
        ],
    },
)
