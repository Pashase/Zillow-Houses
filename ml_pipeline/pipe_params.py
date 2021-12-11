import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.suod import SUOD

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
HBOS_detectors = [HBOS(n_bins='auto', contamination=threshold) for threshold in [0.1, 0.2]]
KNN_detetors = [KNN(n_neighbors=n_neighbors, method='largest') for n_neighbors in [10, 15]]
IForest_detectors = [IForest(n_estimators=150, max_samples='auto', contamination=threshold, random_state=0)
                     for threshold in [0.1, 0.125]]
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
