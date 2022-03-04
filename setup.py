from setuptools import find_packages, setup

setup(
    name="morphelia",
    packages=find_packages(include=["morphelia", "morphelia.*"]),
    version="0.0.3",
    description="Exploratory data analysis for image-based morphological profiling.",
    author="Alexander Marx",
    license="MIT Licence",
    # install_requires=[
    #     line.strip() for line in Path('requirements.txt').read_text('utf-8').splitlines()
    # ],
    entry_points={
        "console_scripts": [
            "StorePlates = morphelia.cli.StorePlates:main",
            "CollectData = morphelia.cli.CollectData:main",
            "Pseudostitch = morphelia.cli.Pseudostitch:main",
            "Aggregate = morphelia.cli.Aggregate:main",
            "CleanData = morphelia.cli.CleanData:main",
            "Normalize = morphelia.cli.Normalize:main",
            "TrainTestSplit = morphelia.cli.TrainTestSplit:main",
            "Subsample = morphelia.cli.Subsample:main",
            "Track = morphelia.cli.Track:main",
        ]
    },
)
