# Report


## Steps 
A the beggining we specified steps and packages we need to use while completing the task.
Main task was to **predict labels for the testing data**
1) [Define the problem](#define)
2) [Data Access](#Data-Access)
3) [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-(EDA))
4) [Data Preprocessing](#Data-Preprocessing)
5) [Model Building](#Model-Building)
6) [Model Validation](#Model-Validation)
7) [Model Execution](#Model-Execution)

<a name="define"></a>
#### Define the problem
Method we use was [**Binary classification**](https://en.wikipedia.org/wiki/Binary_classification)

<a name="Data-Access"></a>
#### Data Access
We acces the data through [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) library

    pandas.DataFrame.read_csv()
and later on to acces about information about DataFrame

    pandas.DataFrame.info() 
    pandas.DataFrame.describe() 
    pandas.DataFrame.shape
    pandas.DataFrame.isnull()
    pandas.DataFrame.value_counts()

<a name="Exploratory-Data-Analysis-(EDA)"></a>
#### Exploratory Data Analysis (EDA) 
Later on we tried to analise the data to learn about it as much as we could using univariate analysis and bivariate analysis crating various [seaborn](https://seaborn.pydata.org/index.html) plots

    seaborn.distplot()
    seaborn.countplot()
    seaborn.boxplot()
    seaborn.scatterplot()
    seaborn.pairplot() 

And to analize distributiom 

    seaborn.histplot())

and correlation

    seaborn.heatmap()

<a name="Data-Preprocessing"></a>
#### Data Preprocessing
At first we splited data to train part and test part using 

    sklearn.model_selection.train_test_split
    
Than we focused on data standarization and normalization using:
    
    sklearn.preprocessing.StandardScaler
    sklearn.pipeline.Pipeline

Trying to understand and set data types, data mixtures, shape, outliers, missing values, noisy data, skewness and kurtosis

<a name="Model-Building"></a>
#### Model Building
We created base line model picking using:
    
    sklearn.model_selection.GridSearchCV
    
in Pipeline, and Hyperopt model picking:

    hpsklearn.HyperoptEstimator
    
to estimate best model

<a name="Model-Validation"></a> 
#### Model Validation
To search for the best model with repeated skf we used:

    sklearn.neighbors.KNeighborsClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.model_selection.RepeatedStratifiedKFold
    sklearn.model_selection.GridSearchCV

using Pipeline, and showing results in classification report building confusion matrix and ploting it

<a name="Model-Execution"></a> 
#### Model Execution
At the end we loaded predicted model and executed prediction in a table showing and the types of incorrect predictions made (what classes incorrect predictions were assigned).

- Precision: A measure of a classifiers exactness.
- Recall: A measure of a classifiers completeness
- F1 Score (or F-score): A weighted average of precision and recall.
