import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metacells", # Replace with your own username
    version="0.0.1",
    author="Pe'er Lab",
    author_email="scp2152@columbia.edu",
    description=" ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    package_dir={'':'src'},
    packages=setuptools.find_packages('src'),
    install_requires=[
        "palantir",
        "scanpy",
        "anndata"
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    include_package_data=True,
    package_data={'': ['src/metacells/data/*', '*.r', '*.R']},
    zip_safe=False
)
