import pandas as pd

import src.data.constants
import src.models.train_model

from joblib import load

from sklearn.metrics import mean_absolute_error


def main():
    test_df = pd.read_csv(src.data.constants.SPLIT_TEST_DATA_PATH)

    X_test = test_df[test_df.columns[~test_df.columns.isin([src.data.constants.TARGET_VAR])]]
    y_test = test_df[src.data.constants.TARGET_VAR]

    best_model = load(src.data.constants.TRAINED_MODEL_PATH)
    y_pred = best_model.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)

    print(f'MAE on test set for randoms forest: {test_mae}')


if __name__ == '__main__':
    main()
