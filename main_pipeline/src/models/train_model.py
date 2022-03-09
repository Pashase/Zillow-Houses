import xgboost as xgb

import pandas as pd

import src.data.constants
from src.data.make_dataset import save_data

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

from joblib import dump, load


def main():
    # read clean - preprocessed data

    data = pd.read_csv(src.data.constants.PATH_TO_SAVE_CLEAN_DF)

    X = data[data.columns[~data.columns.isin([src.data.constants.TARGET_VAR])]]
    y = data[src.data.constants.TARGET_VAR]

    param_grid = {
        'n_estimators': [10, 15, 17],
        'max_depth': [2, 3, 4],
        'min_samples_split': [3, 5, 6, 7, 8],
        'max_features': ['auto', 'sqrt'],
    }

    rf_naive_model = RandomForestRegressor(n_estimators=100, max_depth=4, oob_score=True, n_jobs=-1, random_state=0)

    # 6 folds cross validation using K-fold
    cv_strategy = 6
    search = GridSearchCV(rf_naive_model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv_strategy,
                          verbose=3)

    # split data into train/test (on train cross-validation)
    test_frac = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=0,
                                                        shuffle=True)

    save_data(X_train, y_train, src.data.constants.SPLIT_TRAIN_DATA_PATH)
    save_data(X_test, y_test, src.data.constants.SPLIT_TEST_DATA_PATH)

    # tuning params on train set
    search.fit(X_train, y_train)

    best_rf = search.best_estimator_
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)

    print(f'MAE on test set for random forest: {test_mae}\n'
          'Model is trained!')

    print('Saving the model..')
    dump(best_rf, src.data.constants.TRAINED_MODEL_PATH)
    print(f'Saved to: {src.data.constants.TRAINED_MODEL_PATH}')
    # -------------------------- Gradient Boosting pipeline ---------------------
    # preprocessing_pipe + xgboost model

    # param_grid = {
    #     'xgb_model__n_estimators': [200, 250, 300, 500],
    #     'xgb_model__eta': [0.1, 0.2, 0.3],
    #     'xgb_model__max_depth': [4, 5, 6, 7],
    #     'xgb_model__subsample': [0.7, 0.8],
    # }
    #
    # xgb_model = xgb.XGBRegressor(n_estimators=220, eta=0.2, max_depth=4, subsample=0.7)
    #
    # search = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv_strategy,
    #                       verbose=1)
    #
    # # tuning params on train set
    # search.fit(X_train, y_train)
    #
    # best_xgb_model = search.best_estimator_
    #
    # dm_train = xgb.DMatrix(X_train, label=y_train)
    # dm_test = xgb.DMatrix(X_test, label=y_test)
    #
    # boosting_iterations = 10000
    # watchlist = [(dm_train, 'train'), (dm_test, 'test')]
    # best_xgb_model.train(search.best_params_, dm_train,
    #                      boosting_iterations, watchlist,
    #                      early_stopping_rounds=100, verbose_eval=10)
    #
    # y_pred_xgb = best_xgb_model.predict(dm_test)
    # test_xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    #
    # print(f'MAE on test set for xgboost: {test_xgb_mae}')
    #
    # dump(best_xgb_model, src.data.constants.TRAINED_MODEL_PATH)


if __name__ == '__main__':
    main()
