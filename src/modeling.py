import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_validate)
from sklearn.metrics import classification_report

import config as c
import utils as u


def build(model_name: c.ModelName):
    if model_name == c.ModelName.lgbm:
        model = LGBMClassifier(
            random_state=0,
            verbose=100)

    return model


def train_and_validate(model, X, y):
    X_train, X_val, y_train, y_val\
        = train_test_split(X, y, test_size=0.3, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(classification_report(y_pred, y_val))

    model.fit(X, y)
    u.dump(model, c.PATH_MODEL)
    return model


def train_predict_cv(MLA, X, y, cv=None, X_test=None):
    # create table to compare MLA metrics
    MLA_columns = ['Name', 'Base Estimators', 'Parameters',
                   'Train Score Mean', 'Test Score Mean', 'Test Score 3*STD', 'Time']
    MLA_compare = pd.DataFrame(columns=MLA_columns)
    if hasattr(X, 'columns'):
        feature_importance_compare = pd.DataFrame(index=X.columns)
    else:
        feature_importance_compare = pd.DataFrame()

    if cv is None:
        cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)

    # create table to compare MLA predictions
    if X_test is not None:
        MLA_predictions = {}

    # index through MLA and save performance to table
    row_index = 0
    for idx_mla, alg in enumerate(MLA):

        # set name and parameters
        MLA_name = alg.__class__.__name__
        print(MLA_name)
        MLA_compare.loc[row_index, 'Name'] = MLA_name
        MLA_compare.loc[row_index, 'Parameters'] = str(alg.get_params())

        # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = cross_validate(
            alg, X, y, cv=cv,
            scoring='neg_log_loss',
            return_train_score=True,
            return_estimator=True,
            verbose=100)

        MLA_compare.loc[row_index, 'Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index,
                        'Train Score Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index,
                        'Test Score Mean'] = cv_results['test_score'].mean()
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        # let's know the worst that can happen!
        MLA_compare.loc[row_index,
                        'Test Score 3*STD'] = cv_results['test_score'].std() * 3

        with u.timer('training'):
            alg.fit(X, y)

        # Base estimators of ensemble methods

        ## Stacking in mlxtend
        if hasattr(alg, 'clfs_'):
            MLA_compare.loc[row_index, 'Base Estimators'] = [
                clf.__class__.__name__ for clf in alg.clfs_]

        ## Voting in sklearn
        if hasattr(alg, 'estimators'):
            MLA_compare.loc[row_index, 'Base Estimators'] \
                = [est[0] for est in alg.estimators]

        # Feature importance
        # Feature importance of NGBoost
        if 'NGB' in alg.__class__.__name__ \
                and hasattr(alg, 'feature_importances_'):
            feature_importance_compare[MLA_name] = alg.feature_importances_[0]

        # Feature importance of CatBoost
        elif hasattr(alg, 'get_feature_importance'):
            feature_importance_compare[MLA_name] = alg.get_feature_importance()

        elif hasattr(alg, 'feature_importances_'):
            feature_importance_compare[MLA_name] = alg.feature_importances_

        # Prediction
        if X_test is not None:
            with u.timer('prediction'):
                MLA_predictions.update({
                    MLA_name: {
                        str(MLA_compare.loc[row_index, 'Base Estimators']): {
                            idx_mla: {
                                'predict': alg.predict(X_test),
                                'predict_proba': alg.predict_proba(X_test)
                            }
                        }
                    }
                })

        row_index += 1

    # print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by=['Test Score Mean'],
                            ascending=False, inplace=True)

    if MLA_compare['Base Estimators'].isnull().all():
        MLA_compare.drop('Base Estimators', axis=1, inplace=True)

    if X_test is None:
        return MLA_compare, feature_importance_compare
    return MLA_compare, feature_importance_compare, MLA_predictions
