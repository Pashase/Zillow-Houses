import typing as tp
import pandas as pd
import miceforest as mf

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from joblib import dump, load

target_var = 'logerror'
OUTLIERS_DETECTOR_PATH = 'final_detector.joblib'


class FastMiceTransformer(BaseEstimator, TransformerMixin):
    """
    Fill missing values with mice algo on lightgbm
    """

    def __init__(self, *, imp_kernel_params: dict, mice_params: dict):
        super().__init__()
        self.imp_kernel_params = imp_kernel_params
        self.mice_params = mice_params

    def fit(self):
        return self

    def transform(self, X, y=None):
        print('FastMiceTransformer transform is called')
        data_ = pd.DataFrame(X, y)
        mf_kernel = mf.ImputationKernel(data_, **self.__imp_kernel_params)

        mf_kernel.mice(**self.__mice_params)

        # average all across datasets
        datasets_list = (mf_kernel.complete_data(dataset=ds_idx, iteration=mf_kernel.iteration_count())
                         for ds_idx in range(self.__imp_kernel_params['datasets']))

        # average all across datasets and get the final dataset
        imputed_ds = self.__build_final_imputed_dataset(*datasets_list)

        return imputed_ds

    @staticmethod
    def __build_final_imputed_dataset(*args):
        concatenated_ds = pd.concat([*args])
        by_row_index = concatenated_ds.groupby(concatenated_ds.index)
        final_imputed_ds = by_row_index.mean()

        final_imputed_ds = self.__undummify(final_imputed_ds)

        return final_imputed_ds

    @staticmethod
    def __undummify(df, prefix_sep: str = '_'):
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
    def __init__(self, *, semi_features: list, decimals: int = 0):
        super().__init__()
        self.__semi_features = semi_features
        self.__decimals = decimals

    def fit(self):
        return self

    def transform(self, data):
        print('SemiDtypeFeaturesRounder transform method is called')

        data_ = data.copy()
        how = [decimals] * len(self.__semi_features)
        round_info = dict(zip(self.__semi_features, how))
        rounded_data = data_.round(decimals=round_info)

        # convert dtypes to the most suitable
        rounded_data = rounded_data.convert_dtypes()

        return rounded_data


class DuplicateDetector(BaseEstimator, TransformerMixin):
    """
    Detect and delete duplicate rows
    """

    def fit(self):
        return self

    def transform(self, data):
        print('DuplicateDetector transform method is called')

        data_ = data.copy()

        # get duplicated rows
        duplicated_indexies = data_[data_.duplicated(keep='last')].index
        # drop duplicated rows
        data_.drop(duplicated_indexies, inplace=True)

        return data_


class AnomalyDetector(BaseEstimator, TransformerMixin):
    """
    Anomalies detection by mice on SUOD algorithm witch use some base detectors
    """

    def __init__(self, *, suod_detector_params: dict = None, detector_model_path: str = None):
        super().__init__()
        self.__suod_detector_params = suod_detector_params
        self.__detector_model_path = detector_model_path

    def fit(self):
        return self

    def transform(self, data):
        print('AnomalyDetector transform method is called')

        data_ = data.copy()
        final_detector = None
        if self.__detector_model_path is not None \
                and isinstance(self.__detector_model_path, str) \
                and suod_detector_params is not None:
            # load fitted detector if exist
            final_detector = load(self.__detector_model_path)
        else:
            final_detector = SUOD(**suod_detector_params)

        outliers_labels, confidence = final_detector.predict(data_, return_confidence=True)

        if self.__detector_model_path is not None:
            dump(final_detector, OUTLIERS_DETECTOR_PATH)

        # 1 -- outlier, 0 -- inlier
        outliers_indexies = np.where(outliers_labels == 1)[0]

        # drop outliers by index (copy of the copy)
        after_outliers_data = data_.drop(outliers_indexies, axis=0, inplace=False)

        return after_outliers_data


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Feature engineering step
    """

    def transform(self, data):
        print('FeatureCreator transform method is called')

        data_ = data.copy()
        __feature_engineering(data_)

        return data_

    @staticmethod
    def __feature_engineering(after_outliers_data) -> None:
        after_outliers_data[['latitude', 'longitude']] = after_outliers_data[['latitude', 'longitude']] / 10 ** 6
        after_outliers_data['censustractandblock'] = after_outliers_data['censustractandblock'] / 10 ** 12
        after_outliers_data['rawcensustractandblock'] = after_outliers_data['rawcensustractandblock'] / 10 ** 6

        # Большинство транзакций как можно было видеть выше оcуществляются сразу после выходных
        after_outliers_data['is_monday'] = after_outliers_data['day'].apply(lambda day: 1 if day == 1 else 0).astype(
            'category')

        after_outliers_data['transactiondate'] = pd.to_datetime(
            dict(year=after_outliers_data.year, month=after_outliers_data.month, day=after_outliers_data.day))
        after_outliers_data['is_weekend'] = after_outliers_data['transactiondate'].dt.weekday.apply(
            lambda day: 1 if day in (5, 6) else 0).astype('category')

        # living area proportions
        after_outliers_data['living_area_prop'] = after_outliers_data['calculatedfinishedsquarefeet'] / \
                                                  after_outliers_data[
                                                      'lotsizesquarefeet']
        # tax value ratio
        after_outliers_data['tax_value_ratio'] = after_outliers_data['taxvaluedollarcnt'] / after_outliers_data[
            'taxamount']
        # tax value proportions
        after_outliers_data['tax_value_prop'] = after_outliers_data['structuretaxvaluedollarcnt'] / after_outliers_data[
            'landtaxvaluedollarcnt']

        return None


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Feature transformation step -- log1p
    """

    def __init__(self, *, features_names: list, strategy: tp.Callable):
        super().__init__()
        self.__features_names = features_names
        self.__strategy = strategy

    def fit(self):
        return self

    def transform(self, data):
        print('FeatureTransformer transform method is called')

        data_ = data.copy()
        data_[self.__features_names] = self.__strategy(data_[self.__features_names])

        return data_


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature Selection step by multiple models (Forests, Lasso, RFE on Forests)
    """

    def __init__(self, *, model_selectors: list, select_from_model_params: dict):
        super().__init__()
        self.__model_selectors = model_selectors
        self.__select_from_model_params = select_from_model_params

    def fit(self):
        return self

    def transform(self, data):
        print('FeatureSelector transform method is called')

        data_ = data.copy()
        X = data_[data_.columns[~data_.columns.isin([target_var])]]
        y = data_[target_var]

        selected_features_info = []
        for model in self.__model_selectors:
            model.fit(X, y)

            model_selector = SelectFromModel(model, **self.__select_from_model_params)
            selected_features = data_.columns[model_selector.get_support()].tolist()
            selected_features_info.append(selected_features)

        # select all features selected by all passed in models
        selected_features_by_all_models = list(set().intersection(*selected_features_info))

        after_feature_selection_df = data_[selected_features_by_all_models]

        return after_feature_selection_df


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, steps: list):
        super().__init__()
        self.steps = steps

    def fit(self):
        return self

    def transform(self, X, y=None):
        """
        Preprocessing all data from zero by given preprocessing pipeline
        """

        step_names = (step.__class__.__name__ for step in self.steps)
        steps_info = list(zip(step_names, self.steps))

        preprocessing_pipe = Pipeline(steps_info)

        preprocessed_data = preprocessing_pipe.transform(X)

        return preprocessed_data
