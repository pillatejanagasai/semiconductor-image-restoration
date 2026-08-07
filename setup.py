from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='semiconductor-image-restoration',
    version='1.0.0',
    author='SEMICON India Hackathon 2026 Team',
    description='Semiconductor image restoration using deep learning',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.10',
)
