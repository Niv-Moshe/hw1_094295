import numpy as np
import pandas as pd
from tensorflow import keras
from lstm import used_features, f1_score_mine, read_data, get_filenames, dfs_to_matrix
import sys


def predict(test_directory_path, features_to_use=used_features):
    model = keras.models.load_model('lstm_model.h5', custom_objects={"f1_score_mine": f1_score_mine})
    ########### Test: #############
    print()
    print("Testing...")
    filenames = get_filenames(test_directory_path)
    _, _, test_df_list = read_data(filenames)
    print("Pre-processing")
    x_test, y_test, test_pids = dfs_to_matrix(test_df_list, features_to_use)
    print(f"Test shapes: {x_test.shape=}, {y_test.shape=}")
    del test_df_list
    print("Finished filling missing values in test")
    y_one_hot = []  # one hot vector indicates on label
    for y in y_test:
        if y == 0:
            y_one_hot.append([1, 0])
        else:
            y_one_hot.append([0, 1])
    y_one_hot_test = np.array(y_one_hot)
    y_pred = model.predict(x_test)
    y_preds = [np.argmax(ys) for ys in y_pred]
    y_act = [np.argmax(ys) for ys in y_one_hot_test]

    pred_df = pd.DataFrame(data={'Id': test_pids, 'SepsisLabel': y_preds})
    pred_df.sort_values(by='Id', ascending=True, inplace=True)
    pred_df.to_csv('prediction.csv', index=False, header=False)
    pass


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     raise Exception('Include the test directory as arguments, '
    #                     'e.g., python test.py test_path')
    test_path = '/home/student/HW1/sec_test'  # sys.argv[1]
    predict(test_path)

