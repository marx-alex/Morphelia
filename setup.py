from setuptools import find_packages, setup
from pathlib import Path

setup(
    name='morphelia',
    packages=find_packages(include=['morphelia', 'morphelia.*']),
    version='0.0.2',
    description='Exploratory data analysis for image-based morphological profiling.',
    author='Alexander Marx',
    license='MIT Licence',
    # install_requires=[
    #     line.strip() for line in Path('requirements.txt').read_text('utf-8').splitlines()
    # ],
    entry_points={'console_scripts': [
        'ExpToAD = morphelia.cli.ExpToAD:main',
        'PlateToAD = morphelia.cli.PlateToAD:main',
        'Pseudostitch = morphelia.cli.Pseudostitch:main'
    ]}
)
