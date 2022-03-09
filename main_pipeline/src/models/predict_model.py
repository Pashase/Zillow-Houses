import pandas as pd

import main_pipeline.src.data.constants
import main_pipeline.src.models.train_model

from joblib import load

from sklearn.metrics import mean_absolute_error


def main():
    test_df = pd.read_csv(main_pipeline.src.data.constants.SPLIT_TEST_DATA_PATH)

    X_test = test_df[test_df.columns[~test_df.columns.isin([main_pipeline.src.data.constants.TARGET_VAR])]]
    y_test = test_df[main_pipeline.src.data.constants.TARGET_VAR]

    best_model = load(main_pipeline.src.data.constants.TRAINED_MODEL_PATH)
    y_pred = best_model.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)

    print(f'MAE on test set for random forest model: {test_mae}')
    print(f'Write predictions to {main_pipeline.src.data.constants.PREDICTIONS_PATH}')

    pd.DataFrame(y_pred).to_pickle(main_pipeline.src.data.constants.PREDICTIONS_PATH + '/example_predictions.pkl')

    print('Finished')


if __name__ == '__main__':
    main()
