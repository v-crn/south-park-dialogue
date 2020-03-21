from enum import Enum
from pydantic import BaseModel
from lightgbm import LGBMClassifier

PATH_ROOT = './'
PATH_DATA_DIR = PATH_ROOT + 'data/'
PATH_DATA_RAW_DIR = PATH_DATA_DIR + 'raw/'
PATH_DATA_PROCESSED_DIR = PATH_DATA_DIR + 'processed/'
PATH_MODELS_DIR = PATH_ROOT + 'models/'

PATH_TRAIN = PATH_DATA_RAW_DIR + 'southparklines.zip'
PATH_TRAIN_EXTRA = PATH_DATA_PROCESSED_DIR + 'train_extra.cpkl'
PATH_VECTORIZER = PATH_MODELS_DIR + 'count_vectorizer.cpkl'
PATH_ENCODER = PATH_MODELS_DIR + 'label_encoder.cpkl'
PATH_MODEL = PATH_MODELS_DIR + 'lgbm.cpkl'

PATH_MLA_COMPARE = PATH_MODELS_DIR + 'MLA_compare.csv'
PATH_MLA_FEATURE_IMPORTANCES = PATH_MODELS_DIR + 'MLA_feature_importances.csv'

TARGET = 'Character'


class ModelName(str, Enum):
    lgbm = "lgbm"


class Line(BaseModel):
    Season: int
    Episode: int
    Character: str
    Line: str


# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    # *** Decision Trees Family ***
    # LightGBM
    LGBMClassifier(
        random_state=0,
        verbose=100),

    # # XGBoost: http://xgboost.readthedocs.io/en/latest/model.html
    # XGBClassifier(
    #     objective='multi:softprob',
    #     random_state=0,
    #     verbose=100),

    # # NGBoost
    # NGBClassifier(
    #     # Dist=distns.Bernoulli,
    #     Dist=distns.k_categorical(5),
    #     # Score=LogScore,
    #     # Base=default_tree_learner,
    #     # natural_gradient=True,
    #     # n_estimators=500,
    #     # learning_rate=0.01,
    #     # minibatch_frac=1.0,
    #     # col_sample=1.0,
    #     verbose=100,
    #     # verbose_eval=100,
    #     # tol=1e-4,
    #     random_state=0),

    # # CatBoost
    # # CatBoost relatively takes a long time when training.
    # CatBoostClassifier(
    #     loss_function='Logloss',
    #     early_stopping_rounds=100,
    #     task_type="GPU",
    #     devices='0',
    #     verbose=100,
    #     random_seed=0),

    # ExtraTreeClassifier
    # tree.ExtraTreeClassifier(random_state=0),

    # # Ensemble Methods
    # ensemble.AdaBoostClassifier(random_state=0),
    # # ensemble.BaggingClassifier(random_state=0),
    # # ensemble.GradientBoostingClassifier(random_state=0),
    # ensemble.RandomForestClassifier(
    #     n_estimators=100, criterion='gini', max_depth=None,
    #     min_samples_split=2, min_samples_leaf=1,
    #     min_weight_fraction_leaf=0.0, max_features='auto',
    #     max_leaf_nodes=None, min_impurity_decrease=0.0,
    #     min_impurity_split=None, bootstrap=True, oob_score=False,
    #     n_jobs=None, random_state=0, verbose=0, warm_start=False,
    #     class_weight=None, ccp_alpha=0.0, max_samples=None),

    # # Gaussian Processes
    # gaussian_process.GaussianProcessClassifier(random_state=0),

    # Navies Bayes
    # naive_bayes.BernoulliNB(),
    # naive_bayes.GaussianNB(),

    # # LogisticRegression
    # linear_model.LogisticRegression(
    #     penalty='l2', dual=False, tol=0.0001,
    #     C=1.0, fit_intercept=True, intercept_scaling=1,
    #     class_weight=None,
    #     random_state=0,
    #     solver='lbfgs', max_iter=100, multi_class='auto',
    #     verbose=0, warm_start=False,
    #     n_jobs=None, l1_ratio=None),

    # # SVM
    # svm.SVC(probability=True, random_state=0),
    # svm.SVC(kernel="linear", C=0.025,
    #         probability=True, random_state=0),
    # svm.SVC(gamma='auto', C=1,
    #         probability=True, random_state=0),
    # svm.NuSVC(probability=True, random_state=0),

    # # Nearest Neighbor
    # neighbors.KNeighborsClassifier(),

    # Discriminant Analysis
    # discriminant_analysis.QuadraticDiscriminantAnalysis(),
    # discriminant_analysis.LinearDiscriminantAnalysis(
    #     solver='svd', shrinkage=None, priors=None, n_components=None,
    #     store_covariance=False, tol=0.0001),

    # Neural Network
    # neural_network.MLPClassifier(
    #     hidden_layer_sizes=(100, ), activation='relu',
    #     solver='adam', alpha=0.0001, batch_size='auto',
    #     learning_rate='constant', learning_rate_init=0.001,
    #     power_t=0.5, max_iter=200, shuffle=True,
    #     random_state=0, tol=0.0001, verbose=100,
    #     warm_start=False, momentum=0.9, nesterovs_momentum=True,
    #     early_stopping=True, validation_fraction=0.1,
    #     beta_1=0.9, beta_2=0.999, epsilon=1e-08,
    #     n_iter_no_change=10, max_fun=15000),
]
