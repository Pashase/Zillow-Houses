import miceforest as mf
import numpy as np
import pandas as pd
import typing as tp

import src.data.constants

from joblib import dump, load

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from pyod.models.suod import SUOD
from yellowbrick.model_selection import RFECV


class FastMiceImputer(BaseEstimator, TransformerMixin):
    """
        Fill missing values with mice algo on lightgbm
        """

    def __init__(self, *, imp_kernel_params: tp.Dict, mice_params: tp.Dict):
        super().__init__()
        self.imp_kernel_params = imp_kernel_params
        self.mice_params = mice_params

        self.imp_kernel = None

    def fit(self, X: pd.DataFrame):
        data_ = pd.DataFrame(X)

        if not self.imp_kernel:
            self.imp_kernel = mf.ImputationKernel(data_, **self.imp_kernel_params)

        self.imp_kernel.mice(**self.mice_params)

        return self

    def transform(self, X: pd.DataFrame):
        print('FastMiceImputer transform method is called')
        if not self.imp_kernel:
            raise NotFittedError('This FastMiceImputer instance is not fitted yet. '
                                 "Call 'fit' with appropriate arguments before using this estimator.")
        else:
            assert 'datasets' in self.imp_kernel_params

            # average all across datasets
            datasets_list = (self.imp_kernel.complete_data(dataset=ds_idx, iteration=self.imp_kernel.iteration_count())
                             for ds_idx in range(self.imp_kernel_params['datasets']))

            # average all across datasets and get the final dataset
            imputed_ds = self.__build_final_imputed_dataset(*datasets_list)

            return imputed_ds

    @staticmethod
    def __build_final_imputed_dataset(*args, **kwargs):
        concatenated_ds = pd.concat([*args])
        by_row_index = concatenated_ds.groupby(concatenated_ds.index)
        final_imputed_ds = by_row_index.mean()

        # final_imputed_ds = FastMiceImputer.__undummify(final_imputed_ds)

        return final_imputed_ds

    @staticmethod
    def __undummify(df: pd.DataFrame, prefix_sep: str = '_') -> pd.DataFrame:
        cols2collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col)
                        .idxmax(axis=1)
                        .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                        .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)

        return undummified_df


class SemiDtypeFeaturesRounder(BaseEstimator, TransformerMixin):
    def __init__(self, *, semi_features: tp.List, decimals: int = 0):
        super().__init__()
        self.semi_features = semi_features
        self.decimals = decimals

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        print('SemiDtypeFeaturesRounder transform method is called')

        data_ = pd.DataFrame(X)
        how = [self.decimals] * len(self.semi_features)
        round_info = dict(zip(self.semi_features, how))
        rounded_data = data_.round(decimals=round_info)

        # convert dtypes to the most suitable
        # rounded_data = rounded_data.convert_dtypes()

        return rounded_data


class DuplicateDetector(BaseEstimator, TransformerMixin):
    """
    Detect and delete duplicate rows
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        print('DuplicateDetector transform method is called')

        data_ = pd.DataFrame(X)

        # get duplicated rows
        duplicated_indexies = data_[data_.duplicated(keep='last')].index
        # drop duplicated rows
        data_.drop(duplicated_indexies, inplace=True)

        return data_


class AnomalyDetector(BaseEstimator, TransformerMixin):
    """
    Anomalies detection by mice on SUOD algorithm witch use some base detectors
    """

    def __init__(self, *, suod_detector_params: tp.Dict = None, detector_model_path: str = None):
        super().__init__()
        self.suod_detector_params = suod_detector_params
        self.detector_model_path = detector_model_path

        self.detect_kernel = None

    def fit(self, X: pd.DataFrame):

        if self.detector_model_path and suod_detector_params:
            # load fitted detector if exist
            self.detect_kernel = load(self.detector_model_path)
        else:
            self.detect_kernel = SUOD(**self.suod_detector_params)

        self.detect_kernel.fit(X)

        if self.detector_model_path:
            dump(final_detector, src.data.constants.OUTLIERS_DETECTOR_PATH)

        return self

    def transform(self, X: pd.DataFrame):
        print('AnomalyDetector transform method is called')

        data_ = pd.DataFrame(X)

        outliers_labels, confidence = self.detect_kernel.predict(data_, return_confidence=True)

        # 1 -- outlier, 0 -- inlier
        outliers_indexies = np.where(outliers_labels == 1)[0]

        # drop outliers by index (copy of the copy)
        after_outliers_data = data_.drop(outliers_indexies, axis=0, inplace=True)

        return after_outliers_data


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Feature engineering step
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        print('FeatureCreator transform method is called')

        data_ = pd.DataFrame(X)
        FeatureCreator.__feature_engineering(data_)

        return data_

    @staticmethod
    def __feature_engineering(X: pd.DataFrame) -> None:
        X[['latitude', 'longitude']] = X[['latitude', 'longitude']] / 10 ** 6
        X['censustractandblock'] = X['censustractandblock'] / 10 ** 12
        X['rawcensustractandblock'] = X['rawcensustractandblock'] / 10 ** 6

        # living area proportions
        X['living_area_prop'] = X['calculatedfinishedsquarefeet'] / X['lotsizesquarefeet']
        # tax value ratio
        X['tax_value_ratio'] = X['taxvaluedollarcnt'] / X['taxamount']
        # tax value proportions
        X['tax_value_prop'] = X['structuretaxvaluedollarcnt'] / X['landtaxvaluedollarcnt']

        return None


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Feature transformation step -- log1p
    """

    def __init__(self, *, features_names: tp.List, strategy: tp.Callable):
        super().__init__()
        self.features_names = features_names
        self.strategy = strategy

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        print('FeatureTransformer transform method is called')

        data_ = pd.DataFrame(X)
        print('Data columns before transform:', list(data_.columns))
        data_[self.features_names] = self.strategy(data_[self.features_names], dtype='float')

        return data_


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature Selection step by multiple models (Forests, Lasso, RFE on Forests)
    """

    def __init__(self, *, voices: int, model_selectors: tp.List, select_from_model_params: tp.Dict,
                 rfe_cv_params: tp.Dict):
        super().__init__()
        self.voices = voices
        self.model_selectors = model_selectors
        self.select_from_model_params = select_from_model_params
        self.rfe_cv_params = rfe_cv_params

        self.selection_info = None

    def fit(self, X: pd.DataFrame, y=None):
        assert self.voices <= len(self.model_selectors)

        data_ = pd.DataFrame(X, y)

        self.selection_info = pd.DataFrame(index=data_.columns)

        for model in self.model_selectors:
            model.fit(X, y)

            model_selector = SelectFromModel(model, **self.select_from_model_params)
            self.selection_info[model.__class__.__name__] = model_selector.get_support()

        # recursive feature elimination
        rfe_cv_selector = RFECV(**self.rfe_cv_params)
        rfe_cv_selector.fit(X, y)

        self.selection_info[rfe_cv_selector.__class__.__name__] = rfe_cv_selector.get_support()

        # add a score by counting voices of all selection models
        self.selection_info['Total'] = np.sum(self.selection_info, axis=1)

        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.selection_info is not None:
            data_ = pd.DataFrame(X, y)

            # select features by number of voices of selecting models
            best_features_names = self.selection_info.query(f'Total == {self.voices}').index

            return data_[best_features_names]
        else:
            raise NotFittedError('This FeatureSelector instance is not fitted yet. '
                                 "Call 'fit' with appropriate arguments before using this estimator.")


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, X: pd.DataFrame, y, preprocessing_pipe: Pipeline, *custom_transformers):
        self.X = X
        self.y = y
        self.preprocessing_pipe = preprocessing_pipe
        self.__custom_transformers = custom_transformers

        self.custom_transformer_pipe = None
        self.clear_data = None

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        step_names = (transformer.__class__.__name__ for transformer in self.__custom_transformers)
        steps_info = list(zip(step_names, self.__custom_transformers))

        self.custom_transformer_pipe = Pipeline(steps_info)

        after_transform_df = self.preprocessing_pipe.fit_transform(X)
        self.clear_data = self.custom_transformer_pipe.fit_transform(after_transform_df, y)

        return self.clear_data
