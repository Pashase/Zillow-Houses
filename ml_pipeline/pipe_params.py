import numpy as np
import pandas as pd
import xgboost as xgb
import zillow_pipeline as zp

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import IsolationForest
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
# from pyod.models.copod import COPOD
from pyod.models.suod import SUOD
from pyod.models.rod import ROD

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFECV

from joblib import dump, load

# path main data folder
DATA_PATH = '/Users/pashase_/Downloads/zillow-prize-1'

# loading data
prop_2016 = pd.read_csv(f'{DATA_PATH}/properties_2016.csv')
train_2016_v2 = pd.read_csv(f'{DATA_PATH}/train_2016_v2.csv', parse_dates=['transactiondate'])
sub_example = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')

# test/train datasets
df_train = train_2016_v2.merge(prop_2016, how='left', on='parcelid')

sub_example.rename(columns={'ParcelId': 'parcelid'}, inplace=True)
df_test = sub_example.merge(prop_2016, how='left', on='parcelid')

# parse dates
# df_train['year'] = df_train['transactiondate'].dt.year
df_train['month'] = df_train['transactiondate'].dt.month
df_train['day'] = df_train['transactiondate'].dt.day

# categorical features
categorical_features = [
    'airconditioningtypeid', 'architecturalstyletypeid', 'buildingqualitytypeid',
    'buildingclasstypeid', 'decktypeid', 'fireplaceflag',
    'hashottuborspa', 'heatingorsystemtypeid', 'pooltypeid10',
    'pooltypeid2', 'pooltypeid7', 'propertycountylandusecode',
    'propertylandusetypeid', 'propertyzoningdesc', 'regionidcountry',
    'regionidcity', 'regionidneighborhood', 'storytypeid',
    'storytypeid', 'typeconstructiontypeid', 'taxdelinquencyflag',
    'day', 'month',
]
# numerical features
numerical_features = [
    'parcelid', 'logerror', 'transactiondate', 'basementsqft',
    'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
    'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
    'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
    'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
    'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'latitude',
    'longitude', 'lotsizesquarefeet', 'poolcnt', 'poolsizesum',
    'rawcensustractandblock', 'regionidcounty', 'regionidzip', 'roomcnt',
    'threequarterbathnbr', 'unitcnt', 'yardbuildingsqft17',
    'yardbuildingsqft26', 'yearbuilt', 'numberofstories',
    'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear',
    'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyyear',
    'censustractandblock',
]

# missing values
missing = pd.DataFrame({'percent': df_train.isna().mean()}).sort_values(by='percent', ascending=False)

# work with data, missing percentage < 80% only
features = missing[missing.percent < 0.8].index
considered_features = df_train[features]

# propertyzoningdesc makes data more cardinality because of mane gradations of this feature
df = considered_features.drop(['transactiondate', 'propertyzoningdesc'], axis=1)

# ---------------------- Fast Mice params (Impute step) ----------------------
impute_kernel_params = {'datasets': 3, 'save_all_iterations': True, 'random_state': 0, 'copy_data': False}
mice_params = {'iterations': 13, 'min_sum_hessian_in_leaf': 0.01}

# ---------------------- Semi dtype params ----------------------

semi_features = ['numberofstories', 'garagecarcnt', 'unitcnt', 'calculatedbathnbr', 'fullbathcnt', 'yearbuilt',
                 'regionidzip', 'assessmentyear', 'roomcnt', 'regionidcounty', 'fips', 'bedroomcnt', 'bathroomcnt']
decimals = 0

# ---------------------- Duplicate detector params ----------------------
#
#

# ---------------------- Anomaly detector params ----------------------

LOF_detectors = [LOF(n_neighbors=n_neighbors, algorithm='auto') for n_neighbors in [20, 25]]
HBOS_detectors = [HBOS(n_bins='auto', contamination=treshold) for treshold in [0.1, 0.2]]
KNN_detetors = [KNN(n_neighbors=n_neighbors, method='largest') for n_neighbors in [10, 15]]
IForest_detectors = [IForest(n_estimators=150, max_samples='auto', contamination=treshold, random_state=0)
                     for treshold in [0.1, 0.125]]
# COPOD_detectors = [COPOD(contamination=treshold) for treshold in [0.1, 0.125]]
detector_list = [*LOF_detectors, *HBOS_detectors, *IForest_detectors]
suod_params = {'base_estimators': detector_list, 'n_jobs': -1, 'contamination': 0.1, 'combination': 'average',
               'verbose': False}
# or..

# ---------------------- FeatureTransformer params (log1p) ----------------------

strategy = np.log1p
# features_log_transform = ('garagetotalsqft', 'lotsizesquarefeet',
#                           'finishedsquarefeet12', 'calculatedfinishedsquarefeet',
#                           'structuretaxvaluedollarcnt', 'taxamount',
#                           'taxvaluedollarcnt', 'landtaxvaluedollarcnt',
#                           'living_area_prop', 'tax_value_ratio',
#                           'tax_value_prop',)
selected_num_features_to_log = ['garagetotalsqft', 'lotsizesquarefeet', 'finishedsquarefeet12',
                                'calculatedfinishedsquarefeet', 'structuretaxvaluedollarcnt',
                                'taxamount', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt',
                                'living_area_prop', 'tax_value_ratio', 'tax_value_prop', ]

# ---------------------- FeatureSelector params ----------------------

T = 0.001
max_features_to_select = 85

rf_base = RandomForestRegressor(max_depth=2, oob_score=True, n_jobs=-1, random_state=0)
model_selectors = [
    Lasso(random_state=0),
    rf_base,
    RandomForestRegressor(n_estimators=150, max_depth=3, oob_score=True, n_jobs=-1, random_state=0),
    ExtraTreesRegressor(n_estimators=200, max_depth=3, bootstrap=True, oob_score=True, n_jobs=-1, random_state=0,
                        max_samples=0.8),
]

select_from_model_params = {
    'threshold': T,
    'prefit': True,
    'max_features': max_features_to_select,
}


# -------------------------- Random forest pipeline ---------------------
# preprocessing_pipe + Random Forest model

def get_preprocessed_data(dirty_data, data_transformer: zp.DataTransformer):
    """
    Makes from dirty data clean using DataTransformer
    """
    preprocessed_data = preprocessing_pipe.transform(dirty_data)

    X_clean = preprocessed_data[preprocessed_data.columns[~preprocessed_data.columns.isin([zp.target_var])]]
    y_clean = preprocessed_data[zp.target_var]

    return X_clean, y_clean


def main():
    preprocessing_pipe = zp.DataTransformer(steps=[
        OneHotEncoder(),
        zp.FastMiceTransformer(imp_kernel_params=impute_kernel_params, mice_params=mice_params),
        zp.SemiDtypeFeaturesRounder(semi_features=semi_features, decimals=decimals),
        zp.DuplicateDetector(),
        zp.AnomalyDetector(suod_detector_params=suod_params),
        zp.FeatureCreator(),
        zp.FeatureTransformer(features_names=selected_num_features_to_log, strategy=strategy),
        zp.FeatureSelector(model_selectors=model_selectors, select_from_model_params=select_from_model_params),
        StandardScaler(),
    ])

    X, y = get_preprocessed_data(df, preprocessing_pipe)

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
