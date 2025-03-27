from setuptools import setup, find_packages

setup(
    name="flashback",
    version="0.1.0",
    author="Logan Engstrom",
    description="Flashback: Fused backwards over backwards for attention",
    packages=find_packages(),
    install_requires=[
        'jax[cuda12]',
        'tqdm',
        'dill',
        'pandas',
        'absl-py'
    ]
)

