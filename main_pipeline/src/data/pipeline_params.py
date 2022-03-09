import numpy as np

import main_pipeline.src.data.preprocessing_pipeline as pipe_tfs

from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.iforest import IForest

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

# ---------------------- Fast Mice params (Impute step) ----------------------

impute_kernel_params = {'datasets': 3, 'save_all_iterations': True, 'random_state': 0, 'copy_data': False}
mice_params = {'iterations': 13, 'min_sum_hessian_in_leaf': 0.01}

m_imp = pipe_tfs.FastMiceImputer(imp_kernel_params=impute_kernel_params, mice_params=mice_params)

# ---------------------- Semi dtype params ----------------------

semi_features = ['numberofstories', 'garagecarcnt', 'unitcnt', 'calculatedbathnbr', 'fullbathcnt', 'yearbuilt',
                 'regionidzip', 'assessmentyear', 'roomcnt', 'regionidcounty', 'fips', 'bedroomcnt', 'bathroomcnt']

rounder = pipe_tfs.SemiDtypeFeaturesRounder(semi_features=semi_features)

# ---------------------- Duplicate detector params ----------------------

dupl_detector = pipe_tfs.DuplicateDetector()

# ---------------------- Anomaly detector params ----------------------

LOF_detectors = [LOF(n_neighbors=n_neighbors, algorithm='auto') for n_neighbors in [20, 25]]
HBOS_detectors = [HBOS(n_bins='auto', contamination=threshold) for threshold in [0.1, 0.2]]
KNN_detetors = [KNN(n_neighbors=n_neighbors, method='largest') for n_neighbors in [10, 15]]
IForest_detectors = [IForest(n_estimators=150, max_samples='auto', contamination=threshold, random_state=0)
                     for threshold in [0.1, 0.125]]
detector_list = [*LOF_detectors, *HBOS_detectors, *IForest_detectors]
suod_params = {'base_estimators': detector_list, 'n_jobs': -1, 'contamination': 0.1, 'combination': 'average',
               'verbose': False}

anom_detector = pipe_tfs.AnomalyDetector(suod_detector_params=suod_params)

# ---------------------- FeatureCreator params ----------------------

feature_creator = pipe_tfs.FeatureCreator()

# ---------------------- FeatureTransformer params (log1p) ----------------------

transform_strategy = np.log1p
features_log_transform = ['garagetotalsqft', 'lotsizesquarefeet',
                          'finishedsquarefeet12', 'calculatedfinishedsquarefeet',
                          'structuretaxvaluedollarcnt', 'taxamount',
                          'taxvaluedollarcnt', 'landtaxvaluedollarcnt',
                          'living_area_prop', 'tax_value_ratio',
                          'tax_value_prop', ]
feature_transformer = pipe_tfs.FeatureTransformer(features_names=features_log_transform, strategy=transform_strategy)

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
voices_count = 3
rfe_cv_params = {
    'estimator': rf_base,
    'step': 0.3,
    'min_features_to_select': max_features_to_select,
    'cv': 6,
    'scoring': 'neg_mean_absolute_error',
    'verbose': 5,
    'n_jobs': -1,
}
select_from_model_params = {
    'threshold': T,
    'prefit': True,
    'max_features': max_features_to_select,
}

feature_selector = pipe_tfs.FeatureSelector(voices=voices_count,
                                            model_selectors=model_selectors,
                                            select_from_model_params=select_from_model_params,
                                            rfe_cv_params=rfe_cv_params)
