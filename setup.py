from setuptools import setup, find_packages

setup(
    name="satya-model",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "spacy",
    ],
    package_data={
        '': ['satya/**/*'],  # Include all files in the satya model
    },
)
