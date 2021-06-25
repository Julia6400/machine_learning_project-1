# Steps 
A the beggining we specified steps and packages we need to use while completing the task.
Main task was to **predict labels for the testing data**
1) [Define the problem](#define)
2) [Data Access](#Data)
3) [Exploratory Data Analysis (EDA)](#Exploratory)
4) [Data Preprocessing](#Preprocessing)
5) [Model Building](#Building)
6) [Model Validation](#Validation)
7) [Model Execution](#Execution)

Go directly to [Problems](#problem)

<a name="define"></a>
### Define the problem
Method we use was [**Binary classification**](https://en.wikipedia.org/wiki/Binary_classification)

<a name="Data"></a>
### Data Access
We acces the data through [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) library

    pandas.DataFrame.read_csv()
and later on to acces about information about DataFrame

    pandas.DataFrame.info() 
    pandas.DataFrame.describe() 
    pandas.DataFrame.shape
    pandas.DataFrame.isnull()
    pandas.DataFrame.value_counts()

<a name="Exploratory"></a>
### Exploratory Data Analysis (EDA) 
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

<a name="Preprocessing"></a>
### Data Preprocessing
At first we splited data to train part and test part using 

    sklearn.model_selection.train_test_split
    
Than we focused on data standarization:
    
    sklearn.pipeline.Pipeline
    sklearn.preprocessing.StandardScaler

 and normalization using:
 
    sklearn.preprocessing.MinMaxScaler

We used Univariate Feature Selection 

    sklearn.feature_selection.SelectKBest

to get rid of noisy data
and PCA reshaping data to examine std and get rid of extreme data. 

    sklearn.decomposition.PCA

Later trying to understand and set data types, data mixtures, shape, outliers, missing values, noisy data, skewness and kurtosis by creating heatmaps, pairplots and others

<a name="Building"></a>
### Model Building
We created base line model picking using:
    
    sklearn.model_selection.GridSearchCV

in Pipeline, then among many machine_learning models GridSearchCV pointed out:

    Best model: ExtraTreesClassifier()

and Hyperopt model:

    hpsklearn.HyperoptEstimator
    
to estimate best model, and the best model was:

    Best model: 
    {'learner': 
    
    SVC(C=56.7140159315241, 
    cache_size=512, 
    degree=1, 
    gamma='auto', 
    kernel='linear',
    max_iter=40692963.0, 
    random_state=4, 
    tol=0.00010374472618808929), 
    
    'preprocs': 
    (MinMaxScaler(feature_range=(0.0, 1.0)),), 
    'ex_preprocs': ()}

<a name="Validation"></a> 
### Model Validation
To search for the best model with repeated skf we used:

    sklearn.neighbors.KNeighborsClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.model_selection.RepeatedStratifiedKFold
    sklearn.model_selection.GridSearchCV
    
in Pipeline and to to save model which was:
    
    Best model params: 
    {'classifier': ExtraTreesClassifier(
    min_samples_split=4, 
    n_estimators=117), 
    'classifier__class_weight': None, 
    'classifier__criterion': 'gini',
    'classifier__min_samples_split': 4,
    'classifier__n_estimators': 117}
    
in [binary_clf_model.sav](binary_clf_model.sav)

using Pipeline, and showing results in classification report building:

    sklearn.metrics.confusion_matrix 

and displaying it in plot:
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
 
<a name="Execution"></a> 
### Model Execution


<a name="problem"></a>
# Problems

During programming we came across those problems

1) First: multidimetional dataset - [solution](#First) 
2) Second: Binary classification - [solution](#Second)
3) Third: unbalanced dataset - [solution](#Third)
4) Fourth: basic model - [solution](#Fourth)

<a name="First"></a>
**First** problem we have occured was multidimensional dataset that we loaded from csv files (noticed while calling .info() on data) and the solution we came up with was dimensional reduction by:
- Correlation Heatmap
- Univariate Selection
- PCA
- RFE

Here we used 

    sklearn.preprocessing.StandardScaler
 
for standarization and:

    sklearn.preprocessing.MinMaxScaler

for normalization. Both using Pipeline
    

<a name="Second"></a>
**Second** problem (while calling .describe() on data) was data (binary) classification and we found solutions by using machine learning models like:

    sklearn.linear_model.SGDClassifier
    sklearn.linear_model.LogisticRegression
    sklearn.svm.LinearSVC

<a name="Third"></a>
**Third** problem (while calling .value_count() on data) was that our dataset was unbalanced so the solution was to call:
- Random Undersampling
- [Oversampling / SMOTE](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)

Here we used:

    imblearn.over_sampling.RandomOverSampler
    

<a name="Fourth"></a>
**Fourth** problem was to chose the basic models so we run:

    sklearn.model_selection.GridSearchCV
    hpsklearn.HyperoptEstimator

on various machine learning model to select the best.

