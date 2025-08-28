# setup.py
from setuptools import setup, find_packages

setup(
    name="twin_generator",
    version="0.1.0",
    description="Generate mathematical twin problems via OpenAI agents",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "sympy",
        "matplotlib",
        "pillow",
        "numpy",
        "openai>=1.40.0",
    ],
    entry_points={
        "console_scripts": [
            "twin-generator = twin_generator.cli:main",
        ],
    },
)
