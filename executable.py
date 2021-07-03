from typing import List
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe

import neptune.new as neptune


# Data Access
def data_load() -> List[pd.DataFrame]:
    """
    Load csv files to pandas.DataFrame
    :return: list: of pandas.DataFrame-s
    """
    train_data = pd.read_csv("project_data/train_data.csv", header=None)
    test_data = pd.read_csv("project_data/test_data.csv", header=None)
    train_labels = pd.read_csv("project_data/train_labels.csv", header=None)

    data_df = [train_data, test_data, train_labels]

    return data_df


data = data_load()
names = ["train_data", "test_data", "train_labels"]
df_cols = zip(data, names)


# EDA
def data_info() -> None:
    """
    Shows dfs (list of pandas.DataFrame-s) info
    :return: None
    """

    for d, n in df_cols:
        print(f"""
{20 * '#'}
info for {n}
{20 * '#'}
        """)
        d.info()


def data_describe() -> None:
    """
    Check dataframes with .describe()
    :return: None
    """

    for d, n in df_cols:
        print(f"""
{20 * '#'}
describe for {n}
{20 * '#'}
        """)
        print(d.describe())


def data_isnull() -> None:
    """
    Check dataframes for null values
    :return: None
    """

    for d, n in df_cols:
        print(f"""
{20 * '#'}
descending percentage of null values in columns of {n}
{20 * '#'}
        """)
        print(round(d.isnull().sum().sort_values(ascending=False) / len(d) * 100, 2))


label_data = [data[2]]
label_names = ["train_labels"]


def data_count(df_y: List[pd.DataFrame], col=None) -> None:
    """
    Check labels for balanced/imbalanced dataset"
    :param df_y: list: of pandas.DataFrame-s
    :param col: names of the columns
    :return:
    """

    if col is None:
        col = label_names
    for d, n in zip(df_y, col):
        print(f"""
{20 * '#'}
\nvalues counts of {n}\n
{20 * '#'}
        """)
        print(d.value_counts())


def info():
    data_info()
    data_describe()
    data_isnull()
    data_count(label_data)


# **Neptune**
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Yjc4NDczMC0zNzc1LTQ0ZjEtYTYwYS0wMjNhNjNhMDNiOGEifQ=="
run = neptune.init(project='marcelmilosz/projekt-ml', api_token=NEPTUNE_API_TOKEN)

params = {
    "optimizer": "Marcel"
}
run["parameters"] = params


def send_data_neptune(data, plot_name):
    """ Sending array with data to neptune"""

    for epoch in range(0, len(data)):
        run[plot_name].log(data[epoch])


def single_record(record, record_name):
    """ Sending single record to neptune """

    run[record_name] = record


def stop_run():
    """ Stoping run at the end of the program """

    run.stop()


os.environ["OMP_NUM_THREADS"] = "1"

seed = np.random.seed(147)


def data_load() -> list:
    """
    Loading csv files to DataFrames
    :return: None
    """

    train_data_out = pd.read_csv("project_data/train_data.csv", header=None)
    test_data_out = pd.read_csv("project_data/test_data.csv", header=None)
    train_labels_out = pd.read_csv("project_data/train_labels.csv", header=None)

    # Save to neptune train labels
    a = train_labels_out.values
    tmp = []
    for i in range(0, len(a)):
        tmp.append(int(a[i]))

    send_data_neptune(tmp, "train_labels")

    return [train_data_out, test_data_out, train_labels_out]


train_data, test_data, train_labels = data_load()

train_labels_ravel = train_labels.values.ravel()

# Preprocessing


def pipe_std_minmax(x_1: pd.DataFrame, x_2: pd.DataFrame) -> List[np.array]:
    """
    Data standardization and normalization
    :param x_1: pd.DataFrame: train data
    :param x_2: pd.DataFrame: test data
    :return: list: of np.array-s of standardized train and test data
    """

    pipe = Pipeline([
        ("std", StandardScaler()),
        ("minmax", MinMaxScaler())
    ])

    train_std_minmax_out = pipe.fit_transform(x_1)
    test_std_minmax_out = pipe.fit_transform(x_2)

    return [train_std_minmax_out, test_std_minmax_out]


train_std_minmax, test_std_minmax = pipe_std_minmax(train_data, test_data)

k = int(len(train_data.columns)/3)


def univariate_select(x_1: np.array, x_2: np.array, y_1: np.array, n_of_kbest: int) -> List[np.array]:
    """
    Univariate Selection
    :param x_1: pd.DataFrame: standardized train data
    :param x_2: pd.DataFrame: standardized test data
    :param y_1: np.array: ravel of train labels
    :param n_of_kbest: int: specify number of k best in SelectKBest
    :return: list: of np.array with univariate test and train data
    """

    print(f"Shape before: {x_1.shape}\n")

    test = SelectKBest(score_func=f_classif, k=n_of_kbest)
    fit = test.fit(x_1, y_1)
    features_1 = fit.transform(x_1)
    features_2 = fit.transform(x_2)

    scores = fit.scores_
    score_df = pd.DataFrame(scores, columns=["Scores"])
    print(
        f"Min score: {min(score_df.Scores)}, max score: {max(score_df.Scores)}, mean score: {np.mean(score_df.Scores)}\n")
    print(f"Shape after: {features_1.shape}\n")

    score_df.drop(score_df[score_df.Scores < 1].index, inplace=True)
    l = len(score_df)

    # Save to Neptune
    single_record(min(score_df.Scores), 'univariate_select_min_score')
    single_record(max(score_df.Scores), 'univariate_select_max_score')
    single_record(np.mean(score_df.Scores), 'univariate_select_mean_score')

    if l != n_of_kbest:
        return univariate_select(train_std_minmax, test_std_minmax, train_labels_ravel, l)
    else:
        return [features_1, features_2]


univariate_train, univariate_test = univariate_select(train_std_minmax, test_std_minmax, train_labels_ravel, k)


def pca_select(x_1: np.array, x_2: np.array) -> List[np.array]:
    """
    Principal Component Analysis
    :param x_1: np.array: univariate train data
    :param x_2: np.array: univariate test data
    :return: list: of np.array-s reshaped by PCA test and train data
    """

    print(f"Shape before: {x_1.shape}\n")

    pca = PCA(n_components=100, random_state=seed)
    fit = pca.fit(x_1)
    features_1 = fit.transform(x_1)
    features_2 = fit.transform(x_2)

    print(f"Explained Variance: \n{fit.explained_variance_ratio_}\n")
    print(f"Shape after: {features_1.shape}")

    # Send to Neptune
    send_data_neptune(fit.explained_variance_ratio_, "explained_variance_ration")

    return [features_1, features_2]


pca_train, pca_test = pca_select(univariate_train, univariate_test)


def rfe_select(x_1: np.array, x_2: np.array, y_1: np.array) -> List[np.array]:
    """
    Recursive Feature Elimination
    :param x_1: np.array: reshaped train data
    :param x_2: np.array: reshaped train data
    :param y_1: np.array: ravel of train labels
    :return: list: of np.array-s reshaped by RFE test and train data
    """

    print(f"Shape before: {x_1.shape}\n")

    svc = SVC(kernel="linear", C=1, random_state=seed)
    rfe = RFE(estimator=svc, n_features_to_select=5)
    fit = rfe.fit(x_1, y_1)
    features_1 = fit.transform(x_1)
    features_2 = fit.transform(x_2)

    print(f"Feature Ranking: \n{fit.ranking_}\n")
    print(f"Shape after: {features_1.shape}\n")

    # Send to Neptune
    send_data_neptune(fit.ranking_, "fit-ranking")

    return [features_1, features_2]


rfe_train, rfe_test = rfe_select(pca_train, pca_test, train_labels_ravel)


def random_sampling(x_1: np.array, y_1: np.array) -> List[np.array]:
    """
    Random Oversampling/Undersampling
    :param x_1: np.array: reshaped train data
    :param y_1: np.array: ravel of train labels
    :return: list: of np.array-s after over and undersampling
    """

    over = RandomOverSampler(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    x_resampled_out, y_resampled_out = pipeline.fit_resample(x_1, y_1)

    # Save X and y resampled to neptune

    tmp_X = []
    for i in range(0, len(x_resampled_out)):
        for j in range(0, len(x_resampled_out[i])):
            tmp_X.append(x_resampled_out[i][j])

    send_data_neptune(tmp_X, "X-resampled")
    send_data_neptune(y_resampled_out, "y-resampled")

    return [x_resampled_out, y_resampled_out]


x_resampled, y_resampled = random_sampling(rfe_train, train_labels_ravel)


def save_data(train_x: np.array, test_x: np.array, train_y: np.array) -> None:
    """
    Save to npy file
    :param train_x: np.array: train data after over and undersampling
    :param test_x: np.array: of reshaped test data by RFE
    :param train_y: np.array: train values after over and undersampling
    :return: None
    """

    np.save('project_data/processed_train_X.npy', train_x)
    np.save('project_data/processed_test_X.npy', test_x)
    np.save('project_data/processed_train_y.npy', train_y)

    print("Saving has been completed.")


save_data(x_resampled, rfe_test, y_resampled)

# EDA part 2
df = pd.DataFrame(x_resampled, index=None, columns=None)


def correlation_heatmap(data: pd.DataFrame) -> plt:
    """
    Correlation heatmap for preprocessed train data
    :param data: pd.DataFrame: of train data after over and undersampling
    :return: plt: correlation heatmap
    """

    plt.figure(figsize=(16.9, 8))
    heat_mask = np.triu(np.ones_like(data.corr(), dtype=bool))
    sns.heatmap(data.corr(), mask=heat_mask, vmin=-1, vmax=1, annot=True)
    plt.title("Correlation heatmap for preprocessed train data")

    return plt.show()


correlation_heatmap(df)


def pair_plot(data: pd.DataFrame) -> plt:
    """
    Pair plot for preprocessed train data
    :param data: pd.DataFrame: of train data after over and undersampling
    :return: plt: pair plot for preprocessed train data
    """

    plt.figure(figsize=(14,8))
    g = sns.pairplot(data, corner=True)
    g.fig.suptitle("Pair plot for preprocessed train data")

    return plt.show()


pair_plot(df)


def box_plot(data: pd.DataFrame) -> plt:
    """
    Box plot for preprocessed train data
    :param data: pd.DataFrame: of train data after over and undersampling
    :return: plt: box plot
    """

    plt.figure(figsize=(15.2, 8))
    sns.boxplot(data=data)
    plt.title("Box plot for preprocessed train data")

    return plt.show()


box_plot(df)


def scatter_plot(x_1: np.array, y_1: np.array) -> plt:
    """
    Summarize class distribution and plot scatter for preprocessed train data
    :param x_1: np.array: train data after over and undersampling
    :param y_1: np.array: train values after over and undersampling
    :return: plt: scatter plot for preprocessed data
    """

    counter = Counter(y_1)

    plt.figure(figsize=(15.1, 13))
    for label, _ in counter.items():
        row_ix = np.where(y_1 == label)[0]
        plt.scatter(x_1[row_ix, 0], x_1[row_ix, 1], label=str(label),
                    s=100, marker="o", alpha=0.5, edgecolor="black")
    plt.title(f"Scatter plot for preprocessed data with {counter}")
    plt.legend()

    return plt.show()


scatter_plot(x_resampled, y_resampled)

# Basic Models
X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.25, random_state=seed)


def pipe_basic_model(x_1: np.array, x_2: np.array, y_1: np.array, y_2: np.array) -> None:
    """
    Base line model picking with repeated skf, GridSearchCV in pipeline
    :param x_1: np.array: train of data after over and undersampling
    :param x_2: np.array: test of data after over and undersampling
    :param y_1: np.array: train of values after over and undersampling
    :param y_2: np.array: test of values after over and undersampling
    :return: None
    """

    pipe = Pipeline([("classifier", SVC(kernel="linear", C=1, random_state=seed))])

    search_space = [
        {"classifier": [SVC(random_state=seed)]},
        {"classifier": [LinearSVC(random_state=seed)]},
        {"classifier": [LogisticRegression(random_state=seed)]},
        {"classifier": [KNeighborsClassifier()]},
        {"classifier": [MLPClassifier(random_state=seed)]},
        {"classifier": [AdaBoostClassifier(random_state=seed)]},
        {"classifier": [GradientBoostingClassifier(random_state=seed)]},
        {"classifier": [RandomForestClassifier(random_state=seed)]},
        {"classifier": [ExtraTreesClassifier(random_state=seed)]},
        {"classifier": [AdaBoostClassifier(random_state=seed)]},
        {"classifier": [GaussianNB()]}
    ]

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
    gridsearch = GridSearchCV(pipe, search_space, cv=rskf, verbose=1, n_jobs=-1)
    best_model = gridsearch.fit(x_1, y_1)
    y_pred = best_model.predict(x_2)

    print(f"\nBest model: {best_model.best_estimator_.get_params()['classifier']}")
    print("\nMicro-averaged F1 score on test set: "
          "%0.3f" % f1_score(y_2, y_pred, average='micro'))

    single_record(f1_score(y_2, y_pred, average='micro'), "f1-score")


pipe_basic_model(X_train, X_test, y_train, y_test)


def hyperopt_model(x_1: np.array, x_2: np.array, y_1: np.array, y_2: np.array) -> None:
    """
    Hyperopt model picking
    :param x_1: np.array: train of data after over and undersampling
    :param x_2: np.array: test of data after over and undersampling
    :param y_1: np.array: train of values after over and undersampling
    :param y_2: np.array: test of values after over and undersampling
    :return: None
    """

    model = HyperoptEstimator(
        classifier=any_classifier("cla"),
        preprocessing=any_preprocessing("pre"),
        algo=tpe.suggest,
        max_evals=10,
        trial_timeout=30
    )

    model.fit(x_1, y_1)
    y_pred = model.predict(x_2)

    print(f"\nBest model: {model.best_model()}")
    print("\nMicro-averaged F1 score on test set: "
          "%0.3f" % f1_score(y_2, y_pred, average='micro'))

    send_data_neptune(y_pred, "y_pred")
    single_record(f1_score(y_2, y_pred, average='micro'), "Micro-averaged F1 Score")


hyperopt_model(X_train, X_test, y_train, y_test)


def load_x_npy() -> List[np.array]:
    """
    Load npy to arrays
    :return: list: of np.arrays
    """

    train_x_array_out = np.load('project_data/processed_train_X.npy')
    test_x_array_out = np.load('project_data/processed_test_X.npy')
    train_y_array_out = np.load('project_data/processed_train_y.npy')

    return [train_x_array_out, test_x_array_out, train_y_array_out]


train_x_array, test_x_array, train_y_array = load_x_npy()

X_train, X_test, y_train, y_test = train_test_split(train_x_array, train_y_array, test_size=0.25, random_state=seed)


# Validation
def pipe_skf_grid_model(x_1: np.array, x_2: np.array, y_1: np.array, save: bool = False) -> np.array:
    """
    Search for best model with repeated skf, GridSearchCV and pipeline. Save best model
    :param x_1: np.array: of train split data
    :param x_2: np.array: of test split data
    :param y_1: np.array: of train split values
    :param save: boolean: tels whether to save or not to save best model
    :return: np.array: of predicted model
    """

    pipe = Pipeline([("classifier", KNeighborsClassifier())])

    search_space = [
        {"classifier": [LinearSVC(max_iter=10000, dual=False, random_state=seed)],
         "classifier__penalty": ["l1", "l2"],
         "classifier__C": np.logspace(1, 10, 25),
         "classifier__class_weight": [None, "balanced"]
         },

        {"classifier": [KNeighborsClassifier()],
         "classifier__n_neighbors": np.arange(2, 60, 2),
         "classifier__weights": ["uniform", "distance"],
         "classifier__algorithm": ["auto", "ball_tree", "kd_tree"],
         "classifier__leaf_size": np.arange(2, 60, 2)

         },

        {"classifier": [ExtraTreesClassifier(random_state=seed)],
         "classifier__n_estimators": np.arange(90, 135, 1),
         "classifier__criterion": ["gini", "entropy"],
         "classifier__class_weight": [None, "balanced", "balanced_subsample"],
         "classifier__min_samples_split": np.arange(2, 5, 1)
         }
    ]

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
    gridsearch = GridSearchCV(pipe, search_space, cv=rskf, scoring="f1_micro", verbose=1, n_jobs=-1)
    best_model = gridsearch.fit(x_1, y_1)
    y_pred_out = best_model.predict(x_2)

    print(f"\nBest model params: \n{best_model.best_params_}")
    # UserWarning: One or more of the test scores are non-finite
    print(f"\nModel scorer: \n{best_model.scorer_}")
    print(f"\nModel score: \n{best_model.best_score_}")

    if save:
        filename = "binary_clf_model.sav"
        joblib.dump(best_model, filename)

    single_record(best_model.best_score_, "model_score")

    return y_pred_out


y_pred = pipe_skf_grid_model(X_train, X_test, y_train, save=True)


def clf_report_with_cm(y_true: np.array, y_predicted: np.array) -> None:
    """
    Show classification report. Build confusion matrix and plot it
    :param y_true: np.array: of test split values
    :param y_predicted: np.array: of predicted model
    :return: None
    """

    target_names = ['class -1', 'class 1']
    print(classification_report(y_true, y_predicted, target_names=target_names))

    cm = confusion_matrix(y_true, y_predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    disp.plot()


plt.rcParams["figure.figsize"] = (6, 6)

clf_report_with_cm(y_test, y_pred)


# Execution
def load_predict_model(to_pred):
    """
    Load the model from disk and predict
    :param to_pred: np.array: of test split data
    :return: np.array: of prediction
    """

    filename = "binary_clf_model.sav"
    loaded_model = joblib.load(filename)

    predicted = loaded_model.predict(to_pred)

    return predicted


test_y = load_predict_model(test_x_array)


test_y_df = pd.DataFrame(test_y)
test_y_df.to_csv("project_data/test_labels.csv")

print(test_y_df)
