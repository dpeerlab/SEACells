import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SEACells",
    version="0.1.0",
    author="Pe'er Lab",
    author_email="scp2152@columbia.edu",
    description=" ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dpeerlab/SEACells",
    packages=setuptools.find_packages(),
    install_requires=[
        "palantir",
        "scanpy",
        "anndata",
        "numba>=0.51.2",
        "scipy==1.6"
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    include_package_data=True,
    package_data={'': ['SEACells/data/*','SEACells/Rscripts/*', '*.r', '*.R']},
    zip_safe=False
)
