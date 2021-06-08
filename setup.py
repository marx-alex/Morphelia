from setuptools import find_packages, setup
from pathlib import Path

setup(
    name='morphelia',
    packages=find_packages(include=['morphelia', 'morphelia.*']),
    version='0.0.2',
    description='Python library for analysis of multidimensional morphological data.',
    author='Alexander Marx',
    license='MIT',
    install_requires=[
        line.strip() for line in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    entry_points={'console_scripts': [
        'StoreAD = morphelia.cli.StoreAD:main',
    ]}

)
