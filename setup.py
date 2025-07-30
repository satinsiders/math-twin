# setup.py
from setuptools import setup, find_packages

setup(
    name="twin_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sympy",
        "matplotlib",
        "pillow",
        "numpy",
        "python-dotenv",
        "openai-agents",
    ],
    entry_points={
        "console_scripts": [
            "twin-generator = twin_generator.cli:main",
        ],
    },
)
