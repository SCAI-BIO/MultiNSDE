# MultiNSDEs

## Description

This repository contains the official implementation of **MultiNSDEs**. Full details of the approach are on the [preprint](). 

--------

## Download

To download everything from this repository onto your local directory, execute the following line on your terminal:
```
$ git https://github.com/SCAI-BIO/MultiNSDE.git MultiNSDEs
$ cd MultiNSDEs
```

--------

## Setting the environment

First you need to set an environment. The model has been tested using the "SDGAD" environment. Please create it using

```
$ conda env create -f environment.yml
$ conda activate SDGAD
```

--------

## Data

Please download one of the datasets used in the paper, and preprocess it as described in the supplemental material. It is recommended to organize your data into three separate sets:

1. **Prognosis model data**  
   Data used for training and evaluating the prognosis models.

2. **Causal ML model data**  
   This dataset is essentially the same as in (1), but restricted to patients with **non-missing baseline values**.

3. **Propensity score and observation model data**  
   Data used to train the models required for estimating treatment assignment and observation probabilities.

--------

## Organization

### MultiNSDEs

This folder contains the code to run MultiNSDEs for both prognosis and causal infrence. For a deeper look into the **MultiNSDE** implementation, navigate to the **Networks** folder.


#### MultiNSDEs/Prognosis

This folder contains code to run MultiNSDEs for **prognosis tasks**, including **prediction**, **reconstruction**, and **counterfactual simulations** allowing digital twins generation.  

#### MultiNSDEs/Causal

This folder contains code to run **MultiNSDEs** as outcome models for causal inference.


### MultiNODEs

The **MultiNODEs** folder contains an adapted version of the official [code](https://github.com/SCAI-BIO/MultiNODEs.git) modified to work with the datasets and scenarios described in the paper.


--------

## Run

### Bash scripts

We provide an example showing how to run our model for each dataset, both for the prognosis model and the causal inference approach. For details on the parameters used for each model, please refer to the `configx.sh` file inside each `bas_script/MultiNSDEs/dataset` folder.

To train the model, simply run the `train.sh` script, the model will be trained and validated automatically. If you want to run the model in **extrapolation mode**, just change the `extrapolation` parameter to 1 in the `config1.sh` file and run the `train1.sh` script again, or use directly the `train2.sh`.


--------

## Contact
- [Prof. Dr. Holger Fröhlich](mailto:holger.froehlich@scai.fraunhofer.de)
- [Diego Valderrama](mailto:diego.felipe.valderrama.nino@scai.fraunhofer.de)
- Department of Biomedical AI & Data Science, Fraunhofer Institute for Algorithms and Scientific Computing (SCAI), Schloss Birlinghoven, 1, 53757 Sankt Augustin.
