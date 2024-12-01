import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# data constants
SMALL_BIG_THRESHOLD = 10000
SMALL_DATA_SPLIT = 1 / 3
BIG_DATA_SPLIT = 1 / 10

# classifier constants
RANDOMSTATE = 42
NUMBER_FOLDS = 10
NUMBER_CV_ITER = 10
SCORING = "f1_weighted"

# classifier names constants
CLF_LOGISTIC_REGRESSION = "lr"
CLF_PASSIVE_AGRESSIVE = "pa"
CLF_STOCHASTIC_GRADIENT = "sgd"
CLF_SUPPORT_VECTOR = "svc"
CLF_DECISION_TREE = "dt"
CLF_RANDOM_FOREST = "rf"
CLF_K_NEIGHBORS = "knn"
CLF_GAUSSIAN_NB = "gnb"
CLF_MULTI_PERCEPTRON = "mlp"
CLF_ADA_BOOST = "adb"
CLF_QUADRATIC_DA = "qda"

# pipeline constants
PIPE_ENCODER = "enc"
PIPE_IMPUTER = "imp"
PIPE_TRANSFORMER = "trf"
PIPE_SCALER = "scl"
PIPE_SELECTOR = "fsl"
PIPE_CLASSIFIER = "clf"
PIPE_EMPTY = "passthrough"

CLASSIFIERS = {
    CLF_LOGISTIC_REGRESSION: LogisticRegression(solver="saga", max_iter=500, random_state=RANDOMSTATE),
    CLF_PASSIVE_AGRESSIVE: PassiveAggressiveClassifier(random_state=RANDOMSTATE),
    CLF_STOCHASTIC_GRADIENT: SGDClassifier(random_state=RANDOMSTATE),
    CLF_SUPPORT_VECTOR: SVC(random_state=RANDOMSTATE),
    CLF_DECISION_TREE: DecisionTreeClassifier(random_state=RANDOMSTATE),
    CLF_RANDOM_FOREST: RandomForestClassifier(random_state=RANDOMSTATE),
    CLF_K_NEIGHBORS: KNeighborsClassifier(),
    CLF_GAUSSIAN_NB: GaussianNB(),
    CLF_MULTI_PERCEPTRON: MLPClassifier(max_iter=100, random_state=RANDOMSTATE),
    CLF_ADA_BOOST: AdaBoostClassifier(algorithm="SAMME", random_state=RANDOMSTATE),
    CLF_QUADRATIC_DA: QuadraticDiscriminantAnalysis(),
}

CLASSIFIERS_PARAMETERS = {
    CLF_LOGISTIC_REGRESSION: [
        {
            PIPE_CLASSIFIER: [LogisticRegression()],
            PIPE_CLASSIFIER + "__solver": ["lbfgs", "newton-cg", "newton-cholesky", "sag"],
            PIPE_CLASSIFIER + "__penalty": ["l2"],
            PIPE_CLASSIFIER + "__C": np.logspace(-1, 0.5, 10),
            PIPE_CLASSIFIER + "__max_iter": [500],
            PIPE_CLASSIFIER + "__random_state": [RANDOMSTATE],
        },
        {
            PIPE_CLASSIFIER: [LogisticRegression()],
            PIPE_CLASSIFIER + "__solver": ["saga"],
            PIPE_CLASSIFIER + "__penalty": ["elasticnet"],
            PIPE_CLASSIFIER + "__C": np.logspace(-1, 0.5, 10),
            PIPE_CLASSIFIER + "__max_iter": [500],
            PIPE_CLASSIFIER + "__l1_ratio": [0, 0.5, 1],
            PIPE_CLASSIFIER + "__random_state": [RANDOMSTATE],
        },
        {
            PIPE_CLASSIFIER: [LogisticRegression()],
            PIPE_CLASSIFIER + "__solver": ["liblinear"],
            PIPE_CLASSIFIER + "__penalty": ["l1", "l2"],
            PIPE_CLASSIFIER + "__C": np.logspace(-1, 0.5, 10),
            PIPE_CLASSIFIER + "__max_iter": [500],
            PIPE_CLASSIFIER + "__random_state": [RANDOMSTATE],
        },
    ],
    CLF_SUPPORT_VECTOR: [
        {
            PIPE_CLASSIFIER: [SVC()],
            PIPE_CLASSIFIER + "__kernel": ["linear", "poly", "rbf", "sigmoid"],
            PIPE_CLASSIFIER + "__gamma": ["scale", "auto"],
            PIPE_CLASSIFIER + "__decision_function_shape": ["ovo", "ovr"],
            PIPE_CLASSIFIER + "__random_state": [RANDOMSTATE],
        }
    ],
    CLF_DECISION_TREE: [
        {
            PIPE_CLASSIFIER: [DecisionTreeClassifier()],
            PIPE_CLASSIFIER + "__criterion": ["gini", "entropy", "log_loss"],
            PIPE_CLASSIFIER + "__max_depth": np.arange(3, 100, 1),
            PIPE_CLASSIFIER + "__max_features": [None, "sqrt", "log2"],
            PIPE_CLASSIFIER + "__class_weight": [None, "balanced"],
            PIPE_CLASSIFIER + "__random_state": [RANDOMSTATE],
        }
    ],
}


class DataModel:
    def __init__(self, data, target_name, data_test=None, has_imbalance=False, n_folds=NUMBER_FOLDS):
        self.data = data
        self.target_name = target_name
        self.data_test = data_test
        self.hasImbalance = has_imbalance
        self.n_folds = n_folds

        # split the data into predictor and target sets
        self.X = self.data.drop(self.target_name, axis=1)
        self.y = self.data[self.target_name]

        # stratify cross-validation splits on unevenly distributed target
        k_fold = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOMSTATE)
        strat_k_fold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOMSTATE)
        self.folds = strat_k_fold if self.hasImbalance else k_fold

        # set the train and test data
        self._generate_train_test_split()

    # split the data into train and test sets
    def _generate_train_test_split(self):
        # stratify split on unevenly distributed target
        stratify = self.y if self.hasImbalance else None

        # set the train and test split size depending on the number of observations
        test_size = SMALL_DATA_SPLIT if self.data.shape[0] < SMALL_BIG_THRESHOLD else BIG_DATA_SPLIT

        # split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=stratify, test_size=test_size, random_state=RANDOMSTATE)

    ######################## MODEL TRAINING ########################

    # train classifier pipeline on labeled train data
    def train_model_classifier(self, classifier_name, steps_order, random=False, n_iter=NUMBER_CV_ITER, performance=SCORING):
        # create pipeline structure with empty steps
        pipeline_steps = [
            (PIPE_ENCODER, PIPE_EMPTY),
            (PIPE_IMPUTER, PIPE_EMPTY),
            (PIPE_TRANSFORMER, PIPE_EMPTY),
            (PIPE_SCALER, PIPE_EMPTY),
            (PIPE_SELECTOR, PIPE_EMPTY),
            (PIPE_CLASSIFIER, CLASSIFIERS[classifier_name]),
        ]

        # set parameters of pipeline steps
        pipeline_params = {
            PIPE_ENCODER: self._get_parameters_encoders(),
            PIPE_IMPUTER: self._get_parameters_imputers(),
            PIPE_TRANSFORMER: self._get_parameters_transformers(),
            PIPE_SCALER: self._get_parameters_scalers(),
            PIPE_SELECTOR: self._get_parameters_selectors(),
            PIPE_CLASSIFIER: CLASSIFIERS_PARAMETERS[classifier_name],
        }

        pipeline = Pipeline(pipeline_steps)

        # hypertune parameters for classifier
        for step in steps_order:
            if random:
                clf_cv = RandomizedSearchCV(
                    pipeline,
                    param_distributions=pipeline_params[step],
                    cv=self.folds,
                    scoring=performance,
                    error_score="raise",
                    n_iter=n_iter,
                    random_state=RANDOMSTATE,
                )
            else:
                clf_cv = GridSearchCV(
                    pipeline,
                    param_grid=pipeline_params[step],
                    cv=self.folds,
                    scoring=performance,
                    error_score="raise",
                )

            # check the performance on training data
            clf_cv.fit(self.X_train, self.y_train)
            print(f"tuned {step} score: {clf_cv.best_score_}")

            # save the improved pipeline
            pipeline = clf_cv.best_estimator_

        return pipeline

    # test classifier pipeline on unlabeled test data
    def fit_model_newdata(self, model):
        # predict on unlabeled data
        y_pred = model.predict(self.data_test)

        # create predicted dataset
        data_test_fin = self.data_test.copy()
        data_test_fin[self.target_name] = y_pred
        data_test_fin = data_test_fin.filter([self.target_name])

        return data_test_fin

    ######################## MODEL PARAMETERS ########################

    def _get_parameters_encoders(self):
        return [
            {PIPE_ENCODER: [self._get_categorical_encoder()]},
        ]

    def _get_parameters_imputers(self):
        return [
            {PIPE_IMPUTER: [None]},
            {PIPE_IMPUTER: [SimpleImputer()], PIPE_IMPUTER + "__strategy": ["mean", "median"]},
        ]

    def _get_parameters_transformers(self):
        return [
            {PIPE_TRANSFORMER: [None]},
            {PIPE_TRANSFORMER: [self._get_log_transformer()]},
            # {PIPE_TRANSFORMER: [PowerTransformer()], PIPE_TRANSFORMER + "__standardize": [False], PIPE_TRANSFORMER + "__method": ["yeo-johnson", "box-cox"]},
        ]

    def _get_parameters_scalers(self):
        return [
            {PIPE_SCALER: [None]},
            {PIPE_SCALER: [StandardScaler()]},
            {PIPE_SCALER: [MinMaxScaler()]},
            {PIPE_SCALER: [RobustScaler()]},
        ]

    def _get_parameters_selectors(self):
        return [
            {PIPE_SELECTOR: [None]},
            {PIPE_SELECTOR: [PCA()], PIPE_SELECTOR + "__n_components": np.arange(1, self.X.shape[1], 3), PIPE_SELECTOR + "__random_state": [RANDOMSTATE]},
            {PIPE_SELECTOR: [SelectKBest()], PIPE_SELECTOR + "__k": np.arange(1, self.X.shape[1], 3), PIPE_SELECTOR + "__score_func": [f_classif]},
        ]

    def _get_categorical_encoder(self):
        return ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.X.select_dtypes(exclude=[np.number]).columns),
                ("numerical", PIPE_EMPTY, self.X.select_dtypes(include=[np.number]).columns),
            ]
        )

    def _get_log_transformer(self):
        return ColumnTransformer(
            transformers=[
                ("positive", FunctionTransformer(np.log1p), np.nonzero(self.X.select_dtypes(include=[np.number]).ge(0).all())[0]),
                ("negative", PIPE_EMPTY, np.nonzero(self.X.select_dtypes(include=[np.number]).ge(0).all() == 0)[0]),
            ]
        )

    ######################## MODEL PLOTTING ########################

    # plot classifier performance on training data
    def plot_models_train_performance(self, models=CLASSIFIERS, scale=False, encode=False, performance=SCORING):
        # scale and encode the training predictors if desired
        scaler = MinMaxScaler() if scale else None
        encoder = self._get_categorical_encoder() if encode else None

        results = []
        for model in models.values():
            # create a default pipeline
            pipeline = Pipeline([(PIPE_ENCODER, encoder), (PIPE_SCALER, scaler), (PIPE_CLASSIFIER, model)])

            # check the cross-validated performance on training data
            pipeline.fit(self.X_train, self.y_train)
            cv_results = cross_val_score(pipeline, self.X_train, self.y_train, cv=self.folds, scoring=performance)

            results.append(cv_results)

        plt.figure(figsize=(20, 8))
        plt.boxplot(results, tick_labels=models.keys())
        plt.show()

    # print classifier performance on test data
    def print_models_test_performance(self, models=CLASSIFIERS, scale=False, encode=False):
        # scale and encode the training and test predictors if desired
        scaler = MinMaxScaler() if scale else None
        encoder = self._get_categorical_encoder() if encode else None

        for name, model in models.items():
            # create a default pipeline
            pipeline = Pipeline([(PIPE_ENCODER, encoder), (PIPE_SCALER, scaler), (PIPE_CLASSIFIER, model)])

            # check the performance on test data
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            print(f"{name} - accuracy: {accuracy_score(self.y_test, y_pred)} / " + f"f1: {f1_score(self.y_test, y_pred, average='weighted')}")

    # plot classifier prediction matrix on test data
    def plot_models_test_cmatrix(self, models):
        for name, model in models.items():
            # check the performance on test data
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            # create the confusion matrix of predictions
            cm = confusion_matrix(self.y_test, y_pred)
            classes = self.data[self.target_name].unique()

            plt.figure(figsize=(7, 7))
            plt.title(name)
            sns.heatmap(cm, annot=True, linewidths=0.5, square=True, cmap="Blues", xticklabels=classes, yticklabels=classes)
            plt.ylabel("observed")
            plt.xlabel("predicted")
            plt.show()
