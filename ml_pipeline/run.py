import xgboost as xgb
import zillow_pipeline as zp

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from pipe_params import dirty_data

from pipe_params import impute_kernel_params
from pipe_params import mice_params
from pipe_params import semi_features
from pipe_params import suod_params
from pipe_params import selected_num_features_to_transform
from pipe_params import transform_strategy
from pipe_params import model_selectors
from pipe_params import select_from_model_params


def get_preprocessed_data(data, steps: list):
    """
    Makes from dirty data clean using DataTransformer
    """
    X_dirty = data[data.columns[~data.columns.isin([zp.target_var])]]
    y_dirty = data[zp.target_var]

    step_names = (step.__class__.__name__ for step in steps)
    steps_info = list(zip(step_names, steps))

    preprocessing_pipe = Pipeline(steps_info)
    preprocessed_data = preprocessing_pipe.fit_transform(X_dirty, y_dirty)

    X_clean = preprocessed_data.drop(zp.target_var, axis=1)
    y_clean = preprocessed_data[zp.target_var]

    return X_clean, y_clean


def main():
    preprocessing_steps = [
        OneHotEncoder(),
        zp.FastMiceTransformer(imp_kernel_params=impute_kernel_params, mice_params=mice_params),
        zp.SemiDtypeFeaturesRounder(semi_features=semi_features),
        zp.DuplicateDetector(),
        zp.AnomalyDetector(suod_detector_params=suod_params),
        zp.FeatureCreator(),
        zp.FeatureTransformer(features_names=selected_num_features_to_transform, strategy=transform_strategy),
        zp.FeatureSelector(model_selectors=model_selectors, select_from_model_params=select_from_model_params),
    ]

    X, y = get_preprocessed_data(dirty_data, preprocessing_steps)

    param_grid = {
        'RandomForestRegressor__n_estimators': [100, 150, 170],
        'RandomForestRegressor__max_depth': [2, 3, 4],
        'RandomForestRegressor__min_samples_split': [3, 5, 6, 7, 8],
        'RandomForestRegressor__max_features': ['auto', 'sqrt'],
    }

    rf_naive_model = RandomForestRegressor(n_estimators=100, max_depth=4, oob_score=True, n_jobs=-1, random_state=0)
    rf_naive_pipeline = Pipeline([
        ('RandomForestRegressor', rf_naive_model),
    ])

    # 6 folds cross validation using K-fold
    cv_strategy = 6
    search = GridSearchCV(rf_naive_pipeline, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv_strategy,
                          verbose=3)

    # split data into train/test (on train cross-validation)
    test_frac = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=0,
                                                        shuffle=True)
    # tuning params on train set
    search.fit(X_train, y_train)

    best_rf = search.best_estimator_
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)

    print(f'MAE on test set for randoms forest: {test_mae}')

    # -------------------------- Gradient Boosting pipeline ---------------------
    # preprocessing_pipe + xgboost model

    param_grid = {
        'xgb_model__n_estimators': [200, 250, 300, 500],
        'xgb_model__eta': [0.1, 0.2, 0.3],
        'xgb_model__max_depth': [4, 5, 6, 7],
        'xgb_model__subsample': [0.7, 0.8],
    }

    xgb_model = xgb.XGBRegressor(n_estimators=220, eta=0.2, max_depth=4, subsample=0.7)

    xgb_pipeline = Pipeline([
        #     ('pca', PCA()),
        ('xgb_model', xgb_model),
    ])

    search = GridSearchCV(xgb_pipeline, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv_strategy,
                          verbose=1)

    # tuning params on train set
    search.fit(X_train, y_train)

    best_xgb_model = search.best_estimator_

    dm_train = xgb.DMatrix(X_train, label=y_train)
    dm_test = xgb.DMatrix(X_test, label=y_test)

    boosting_iterations = 10000
    watchlist = [(dm_train, 'train'), (dm_test, 'test')]
    best_xgb_model.train(search.best_params_, dm_train,
                         boosting_iterations, watchlist,
                         early_stopping_rounds=100, verbose_eval=10)

    y_pred_xgb = best_xgb_model.predict(dm_test)
    test_xgb_mae = mean_absolute_error(y_test, y_pred_xgb)

    print(f'MAE on test set for xgboost: {test_xgb_mae}')

    return None


if __name__ == '__main__':
    main()
