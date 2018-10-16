import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="supportr",
    version="0.1",
    author="Zijian Wang and David Jurgens",
    author_email="jurgens@umich.edu",
    description="Supportr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=["numpy", "csv", "datrie", "nltk", "gensim",
                      "pandas", "spacy", "sklearn"],
    url="https://github.com/davidjurgens/support",
    include_package_data=True,
    packages=setuptools.find_packages()

)
