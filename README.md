SEACells: 
------

**S**ingle-c**E**ll **A**ggregation for High Resolution **Cell S**tates 

#### Installation and dependencies
1. SEACells has been implemented in Python3.8 can be installed from source: 
		
		$> git clone https://github.com/dpeerlab/SEACells.git
		$> cd SEACells
		$> python setup.py install
 
2. If you are using `conda`, you can use the `environment.yaml` to create a new environment and install SEACells.
```
conda env create -n seacells --file environment.yaml
conda activate seacells
```

3. You can also use `pip` to install the requirements 
```
pip install -r requirements.txt
```
And then follow step (1)

4. MulticoreTSNE issues can be solved using 
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


1. <b>ATAC preprocessing</b>
`notebooks/ArchR` folder contains the preprocessing scripts and notebooks including peak calling using NFR fragments. See notebook [here](https://github.com/dpeerlab/SEACells/blob/main/notebooks/ArchR/ArchR-preprocessing.ipynb) to get started 

2. <b>Computing SEACells</b>
A tutorial on SEACells usage and results visualization for single cell data can be found in the [SEACell computation notebook] (https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_computation.ipynb).

3. <b>Gene regulatory toolkit</b>
Peak gene correlations, gene scores and gene accessibility scores can be computed using the [ATAC analysis notebook] (https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_ATAC_analysis.ipynb).

4. <b>Large-scale data integration using SEACells </b>
Details are avaiable in the [COVID integration notebook] (https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_COVID_integration.ipynb)


5. <b>Cross-modality integration </b>
Integration between scRNA and scATAC can be performed following the [Integration notebook](https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_domain_adapt.ipynb)

#### Citations

Coming soon!
____

Release Notes
-------------

