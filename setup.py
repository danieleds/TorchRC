from setuptools import setup

# read the contents of the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="torch_rc",
    version="0.2.3",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "torch>=1.8",
    ],
    python_requires='>=3.8',
    packages=['torch_rc'],
)
