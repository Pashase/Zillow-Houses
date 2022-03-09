import pandas as pd
import typing as tp

import main_pipeline.src.data.constants
import main_pipeline.src.data.pipeline_params
import main_pipeline.src.data.preprocessing_pipeline as pipe_tfs

from functools import wraps
from sklearn.pipeline import Pipeline


def load_data(prop_data_path: str,
              train_ds_path: str,
              sample_sub_path: str,
              **kwargs) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # loading data
    prop_2016 = pd.read_csv(prop_data_path)
    train_2016_v2 = pd.read_csv(train_ds_path, parse_dates=['transactiondate'])
    sub_example = pd.read_csv(sample_sub_path)

    return prop_2016, train_2016_v2, sub_example


def save_data(X: pd.DataFrame, y: pd.Series, path: str) -> None:
    data_ = X.copy()
    data_[main_pipeline.src.data.constants.TARGET_VAR] = y.copy()

    data_.to_csv(path, index=False)

    return None


def make_filtering(miss_percent: float = .8,
                   exclude_features: tp.Tuple = ('transactiondate', 'propertyzoningdesc', 'parcelid')) -> tp.Callable:
    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # get pre dirty data (after reading)
            df_dirty = func(*args, **kwargs)

            # parse transactiondate
            if 'transactiondate' in exclude_features:
                # df_train['year'] = df_train['transactiondate'].dt.year
                df_dirty['month'] = df_dirty['transactiondate'].dt.month
                df_dirty['day'] = df_dirty['transactiondate'].dt.day

            # missing values
            missing = pd.DataFrame({'percent': df_dirty.isna().mean()})

            # work with data, missing percentage < 80% only
            features = missing[missing.percent < miss_percent].index

            df_dirty = df_dirty[features]

            # propertyzoningdesc makes data more cardinality because of many gradations of this feature
            # transactiondate is parsed -> delete

            df_dirty = df_dirty.drop(list(exclude_features), axis=1)

            return df_dirty

        return wrapper

    return decorator


def convert_dtypes(categorical_names: tp.Tuple) -> tp.Callable:
    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            df_dirty = func(*args, **kwargs)

            assert isinstance(df_dirty, pd.DataFrame)

            # categorical converting
            for feature_name in df_dirty.columns:
                if feature_name in categorical_names:
                    df_dirty[feature_name] = df_dirty[feature_name].astype('category')
            # another types
            # df_dirty = df_dirty.convert_dtypes()

            return df_dirty

        return wrapper

    return decorator


@convert_dtypes(categorical_names=main_pipeline.src.data.constants.CATEGORICAL_FEATURES_NAMES)
@make_filtering(miss_percent=.8)
def get_dirty_data(prop_2016: pd.DataFrame,
                   train_2016_v2: pd.DataFrame,
                   **kwargs) -> pd.DataFrame:
    prop_2016_c, train_2016_v2_c = prop_2016.copy(), train_2016_v2.copy()

    # pre-dirty data
    df_dirty = train_2016_v2_c.merge(prop_2016_c, how='left', on='parcelid')

    return df_dirty


def main() -> None:
    prop_2016, train_2016_v2, sub_example = load_data(main_pipeline.src.data.constants.PROP_DATA_PATH,
                                                      main_pipeline.src.data.constants.TRAIN_DATA_PATH,
                                                      main_pipeline.src.data.constants.SAMPLE_SUB_DATA_PATH)
    df_dirty_train = pd.get_dummies(get_dirty_data(prop_2016, train_2016_v2)[1000:1180])

    X_dirty = df_dirty_train[
        df_dirty_train.columns[~df_dirty_train.columns.isin([main_pipeline.src.data.constants.TARGET_VAR])]
    ]
    y_dirty = df_dirty_train[main_pipeline.src.data.constants.TARGET_VAR]

    prep_pipe = Pipeline(steps=[
        ('mice imputer', main_pipeline.src.data.pipeline_params.m_imp),
        ('rounder', main_pipeline.src.data.pipeline_params.rounder),
        ('duplicate detector', main_pipeline.src.data.pipeline_params.dupl_detector),
        ('anomaly detector', main_pipeline.src.data.pipeline_params.anom_detector),
        ('feature creator', main_pipeline.src.data.pipeline_params.feature_creator),
        ('feature transformer', main_pipeline.src.data.pipeline_params.feature_transformer),
    ])

    transformers = (main_pipeline.src.data.pipeline_params.feature_selector,)

    data_transformer = pipe_tfs.DataTransformer(X_dirty, y_dirty, prep_pipe, *transformers)

    # get clean X
    X_clean = data_transformer.fit_transform(X_dirty, y_dirty)
    # y
    y_clean = y_dirty.copy()

    X_clean[main_pipeline.src.data.constants.TARGET_VAR] = y_clean

    # serialize processed data
    X_clean.to_pickle(main_pipeline.src.data.constants.PROCESSED_DATA_PATH + '/processed_df.pkl')

    return None


if __name__ == '__main__':
    main()
