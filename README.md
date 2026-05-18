# MultiNSDEs

## Description

This repository contains the official implementation of **MultiNSDEs**. Full details of the approach are on the [preprint](). 


## Download

To download everything from this repository onto your local directory, execute the following line on your terminal:
```
$ git https://github.com/SCAI-BIO/MultiNSDE.git MultiNSDEs
$ cd MultiNSDEs
```

## Setting the environment

First you need to set an environment. The model has been tested using the "SDGAD" environment. Please create it using

```
$ conda env create -f environment.yml
$ conda activate SDGAD
```

## Data

Please download one of the datasets used in the paper, preprocess it as described in the supplemental material, and save the files in a folder structured as:
> `data/A4/`

## Run

### Bash scripts

We provide an example showing how to run our model for each dataset, both for the prognosis model and the causal inference approach. For details on the parameters used for each model, please refer to the `config.sh` file inside each `bas_script/` folder.

To train the model, simply run the `train` script, the model will be trained and validated automatically. If you want to run the model in **extrapolation mode**, just change the `extrapolation` parameter in the `config.sh` file and run the `train` script again.

For a deeper look into the **MultiNSDE** implementation, navigate to its folder and follow the instructions in the README for more details.

The **MultiNODEs** folder contains an adapted version of the official [code](https://github.com/SCAI-BIO/MultiNODEs.git) modified to work with the datasets and scenarios described in the paper.


--------
## Contact
- [Prof. Dr. Holger Fröhlich](mailto:holger.froehlich@scai.fraunhofer.de)
- [Diego Valderrama](mailto:diego.felipe.valderrama.nino@scai.fraunhofer.de)
- Department of Biomedical AI & Data Science, Fraunhofer Institute for Algorithms and Scientific Computing (SCAI), Schloss Birlinghoven, 1, 53757 Sankt Augustin.
