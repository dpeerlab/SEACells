SEACells: 
------

**S**ingle-c**E**ll **A**ggregation for High Resolution **Cell S**tates 

#### Installation and dependencies
1. SEACells has been implemented in Python3 can be installed from source: 
		
		$> git clone https://github.com/dpeerlab/SEACells.git
		$> cd SEACells
		$> python setup.py install
 
2. If you are using `conda`, the following commands can be used to solve any issues with MultiCoreTSNE
```
conda create --name seacells -c conda-forge -c bioconda cython python=3.8
conda activate seacells
pip install git+https://github.com/settylab/Palantir@removeTSNE
git clone https://github.com/dpeerlab/SEACells.git
cd SEACells 
python setup.py install
```


3. SEACells depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`.

    All the dependencies will be automatically installed using the above commands

4. To uninstall:
		
		$> pip uninstall SEACells

#### Usage

A tutorial on SEACells usage and results visualization for single cell RNA-seq data can be found in `notebooks/example_notebook.ipynb`.

#### Citations

____

Release Notes
-------------

