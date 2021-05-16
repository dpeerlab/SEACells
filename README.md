Metacells
------

todo

#### Installation and dependencies
1. Metacells has been implemented in Python3 and can be installed from source by running :

        $> git clone https://github.com/dpeerlab/metacells.git
 then navigating to the installation folder `metacells` and running 
 $> python setup.py install
 
2. Metacells depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`

    All the dependencies will be automatically installed using the above commands

3. To uninstall:
		
		$> pip uninstall metacells

#### Usage

A tutorial on Metacells usage and results visualization for single cell RNA-seq data can be found in this notebook: 

TODO

#### Processed data and metadata
Sample ```scanpy anndata``` objects can be loaded using the ```utils.load_data()``` function. 
Each object has the following elements
* todo 
* `.X`: Filtered, normalized and log transformed count matrix 
* `.raw`: Filtered raw count matrix
* `.obsm['Metacell']`: Assignments to metacells



#### Citations


____

Release Notes
-------------

