# Machine learning project
#### Dawid Jaskulski, Marcel Miłosz, Łukasz Kowalczyk


Graduation project for machine learning classes

## TASK

- The data consists of 2 splits train_data.csv and test_data.csv. There are numeric observations labelled either -1 or +1. Labels for the training data are located in in train_labels.csv. 
- The aim of the project is to predict labels for the testing data.
- Save predictions in test_labels.csv
- Prepare a report (saved as report.md)with the explanations on how you came up with the solution. What obstacles you faced and how did you solve them. Add also information about data quality and so on.

## RULES
- PEP8
- DRY
- docstring
- type hints
- every task is a new branch with code review before merge
- Integraton with: https://neptune.ai/

## Data 

All data are in directory
```sh
project_data
```
they are not included in git because of the large memory usage data can be downloaded [here](https://drive.google.com/drive/folders/1We8Z_pjXJmnrtrPrAkHMU9V1hL3wFCKu?usp=sharing)

## Installation

All machine_learning_project packages are specified in [environment.yml](https://github.com/djaskulski/machine_learning_project/blob/main/environment.yml)

Recomended eviroment for the project is [conda](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html)

Simply type in:
```sh
conda env create -f environment.yml
```

