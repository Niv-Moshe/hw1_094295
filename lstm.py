import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from keras.layers import LSTM, Dense, Bidirectional
from keras.models import Sequential
from keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives, BinaryCrossentropy
from keras.regularizers import L2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras_tuner import Hyperband, RandomSearch, Objective
from hyperopt import hp
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import os
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")
tf.config.run_functions_eagerly(True)

random.seed(42)
np.random.seed(42)

# list out lab test features for imputation
labs = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
        'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
        'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
# list out vital signal features for imputation
vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
# list out demographic features for imputation
demogs = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
# labels
labels = ['SepsisLabel']
nonlabels = vitals + labs + demogs

unwanted_features = ['Bilirubin_direct', 'Fibrinogen', 'TroponinI']

used_features = [col for col in nonlabels if col not in unwanted_features]



def get_filenames(directory_path):
    filepath = Path(directory_path)
    filenames = [fname for fname in filepath.iterdir() if fname.is_file() and fname.suffix == '.psv']  # [:100]
    return filenames


def read_data(filenames):
    all_df = []
    sick = []
    healthy = []
    for filename in tqdm(filenames):
        with filename.open() as fp:
            patient_id = str(filename).split("_")[1].split(".")[0]
            df = pd.read_csv(fp, sep='|')
            df['pid'] = len(df) * [int(patient_id)]
            if 1 in list(df['SepsisLabel']):
                ind = list(df['SepsisLabel']).index(1)
                df = df[:ind + 1]
                df['SepsisLabel'] = len(df) * [1]
                sick.append(df)
            else:
                healthy.append(df)
            all_df.append(df)
    return sick, healthy, all_df


# function to fill missing values
def impute_missing_vals(df, attributes):
    df_clean = df.copy()
    for att in attributes:
        if df_clean[att].isnull().sum() == len(df_clean):
            df_clean[att] = df_clean[att].fillna(0)
        elif df_clean[att].isnull().sum() == len(df_clean) - 1:
            df_clean[att] = df_clean[att].ffill().bfill()
        else:
            df_clean[att] = df_clean[att].interpolate(method='nearest', limit_direction='both')
            df_clean[att] = df_clean[att].ffill().bfill()
    return df_clean


def dfs_to_matrix(df_list, features_to_use):
    x, y = [], []
    pids = []
    all_lengths = [len(patient_df) for patient_df in df_list]
    max_len = np.max(all_lengths)
    for patient_df in tqdm(df_list):
        sepsis_labels = patient_df['SepsisLabel']
        patient_df.drop(['SepsisLabel'], axis=1, inplace=True)
        pids.append(patient_df['pid'].to_list()[0])
        patient_df.drop(['pid'], axis=1, inplace=True)
        patient_df_use = patient_df[features_to_use].copy(deep=True)
        del patient_df
        patient_df_clean = impute_missing_vals(patient_df_use, list(patient_df_use.columns))
        # print(f"{len(patient_df.columns)=}")
        y.append(int(sepsis_labels.sum() > 0))
        length = len(patient_df_clean)
        if length >= max_len:
            x.append(patient_df_clean.values)
        else:  # padding with zeros
            temp_df = pd.DataFrame(dict(zip(patient_df_clean.columns, np.zeros((len(patient_df_use.columns),
                                                                                max_len - length), dtype=np.float32))))
            temp_df = pd.concat([temp_df, patient_df_clean], axis=0)
            x.append(temp_df.values)
    # numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y, pids


def f1_score_mine(y_true, y_pred):
    fn_func = FalseNegatives()
    fn = float(fn_func(y_true, y_pred))
    fp_func = FalsePositives()
    fp = float(fp_func(y_true, y_pred))
    tp_func = TruePositives()
    tp = float(tp_func(y_true, y_pred))
    f1 = (2 * tp) / float(2 * tp + fp + fn)
    return f1


def build_model(hp, features_to_use=used_features):
    # hp means hyper parameters
    num_of_neurons_lstm = hp.Choice('num_of_neurons_lstm', values=[32, 64, 128, 256])
    l2 = hp.Choice('l2', values=[0.0, 1e-2, 1e-3, 1e-4])
    num_of_neurons_dense = hp.Choice('num_of_neurons_dense', values=[32, 64, 128])
    model = Sequential()
    model.add(LSTM(units=num_of_neurons_lstm, kernel_regularizer=L2(l2), return_sequences=True,
                   input_shape=(None, len(features_to_use))))
    model.add(LSTM(units=num_of_neurons_lstm, kernel_regularizer=L2(l2)))
    # providing range for number of neurons in a hidden layer
    model.add(Dense(units=num_of_neurons_dense, kernel_regularizer=L2(l2), activation='relu'))
    # output layer
    model.add(Dense(units=2, kernel_regularizer=L2(l2), activation='softmax'))

    # compiling the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[f1_score_mine, 'AUC', 'accuracy'])
    return model


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # feeding the model and parameters to Random Search
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    print("Random Search...")
    tuner = RandomSearch(build_model,
                         objective=Objective('val_f1_score_mine', direction='max'), # 'val_accuracy',
                         metrics=[f1_score_mine],
                         max_trials=5,
                         executions_per_trial=3,
                         directory='tuner1',
                         project_name='Sepsis')
    # print("Hyperband...")
    # tuner = Hyperband(build_model,
    #                      objective=Objective('val_f1_score_mine', direction='max'),
    #                      max_epochs=5,
    #                      factor=3,
    #                      directory='tuner1',
    #                      project_name='Sepsis')
    tuner.search_space_summary()
    print()
    print("Starting search...")
    tuner.search(X_train, y_train, epochs=3, validation_data=(X_val, y_val), callbacks=[stop_early])
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return tuner, best_hps


def my_hyperparameter_tuning(X_train, y_train, X_val, y_val, features_to_use=used_features):
    num_of_neurons_lstm = [64, 128, 256]
    num_of_neurons_dense = [32, 64, 128]
    l2_regs = [0.0, 1e-3, 1e-4]
    combs = product(num_of_neurons_lstm, num_of_neurons_dense, l2_regs)
    all_models_f1_scores = []   # a tuple of trained model, comb dict, model results and val f1 score
    count = 0
    for comb in tqdm(combs):
        units_lstm = comb[0]
        units_dense = comb[1]
        l2 = comb[2]
        if units_dense > units_lstm:
            continue
        if count <= 12:
            count += 1
            continue
        print(f"Continue hyper parameter from comb number {count}")
        count += 1

        params = {'num_of_neurons_lstm': units_lstm, 'num_of_neurons_dense': units_dense, 'l2': l2}
        print()
        print(f"Current params = {params}")
        model = Sequential()
        model.add(LSTM(units=units_lstm, kernel_regularizer=L2(l2),
                       return_sequences=True, input_shape=(None, len(features_to_use))))
        model.add(LSTM(units=units_lstm, kernel_regularizer=L2(l2)))
        model.add(Dense(units=units_dense, kernel_regularizer=L2(l2),
                        activation='relu'))
        model.add(Dense(units=2, kernel_regularizer=L2(l2), activation='softmax'))
        # model.summary()

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=[f1_score_mine, 'AUC', 'accuracy'])
        hist = model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1,
                         validation_data=(X_val, y_val), shuffle=True)
        val_f1 = hist.history['f1_score_mine']
        all_models_f1_scores.append((model, params, hist.history, val_f1))
    best = sorted(all_models_f1_scores, key=lambda tup: tup[3], reverse=True)[0]  # descending
    best_model = best[0]
    best_model.save('lstm_model.h5')
    best_params = best[1]
    model_results = best[2]
    print(f"{best_params=}")
    print(f"{model_results=}")
    return best_model, best_params, model_results


def split_train_val(sick_X, healthy_X):
    X_train_sick, X_val_sick = train_test_split(sick_X, test_size=0.10, random_state=42)
    num_sick_train = len(X_train_sick)
    num_sick_val = len(X_val_sick)
    del sick_X
    X_train_healthy, X_val_healthy = train_test_split(healthy_X, test_size=0.10, random_state=42)
    del healthy_X
    X_train = np.vstack((X_train_sick, X_train_healthy))
    y_train = np.array(list(np.ones(X_train_sick.shape[0], dtype=int)) +
                       list(np.zeros(X_train_healthy.shape[0], dtype=int)))
    del X_train_sick
    del X_train_healthy
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)  # shuffle train
    X_train = X_train[arr]
    y_train = y_train[arr]
    X_val = np.vstack((X_val_sick, X_val_healthy))
    y_val = np.array(list(np.ones(X_val_sick.shape[0], dtype=int)) +
                     list(np.zeros(X_val_healthy.shape[0], dtype=int)))
    del X_val_sick
    del X_val_healthy
    print(len(X_train), "samples in train split-", round(100*num_sick_train/len(X_train), 2), "percent sick")
    print(len(X_val), "samples in val split-", round(100*num_sick_val/len(X_val), 2), "percent sick")
    return X_train, y_train, X_val, y_val


def train(features_to_use=used_features, make_matrix=True):
    print(f"Using: {len(features_to_use)} columns")
    ########### Train: #############
    sick, healthy, train_df_list = read_data(get_filenames('data/train'))
    print("Got", len(train_df_list), "samples from train:", len(sick), "sick and", len(healthy), "healthy")
    del sick
    del healthy
    # Full data:
    print("Pre-processing...")
    if make_matrix:
        x_train, y_train, train_pids = dfs_to_matrix(train_df_list, features_to_use)
        # np.save('lstm_data/x_train.npy', x_train)
        # np.save('lstm_data/y_train.npy', y_train)
        # np.save('lstm_data/train_pids.npy', train_pids)
    else:
        print(f"Loading lstm data")
        # x_train = np.load('lstm_data/x_train.npy')
        # y_train = np.load('lstm_data/y_train.npy')
        # train_pids = np.load('lstm_data/train_pids.npy')


    print(f"Train shapes: {x_train.shape=}, {y_train.shape=}")
    del train_df_list
    print("Finished filling missing values in train")
    sick_index = []
    healthy_index = []
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sick_index.append(i)
        else:
            healthy_index.append(i)
    x_sick = x_train[sick_index]
    x_healthy = x_train[healthy_index]
    print(f"Splitting train to train and val...")
    # x_train, y_train, x_val, y_val = split_train_val(x_sick, x_healthy)
    del x_sick
    del x_healthy
    del sick_index
    del healthy_index

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    print(f"{x_train.shape=}")
    print(f"{y_train.shape=}")
    print(f"{x_val.shape=}")
    print(f"{y_val.shape=}")

    y_one_hot = []  # one hot vector indicates on label
    for y in y_train:
        if y == 0:
            y_one_hot.append([1, 0])
        else:
            y_one_hot.append([0, 1])
    y_one_hot_train = np.array(y_one_hot)
    del y_train

    y_one_hot = []  # one hot vector indicates on label
    for y in y_val:
        if y == 0:
            y_one_hot.append([1, 0])
        else:
            y_one_hot.append([0, 1])
    y_one_hot_val = np.array(y_one_hot)
    del y_val
    ########### Models: #############
    # print(f"Running my hyper-parameter tuning")
    # best_model, best_hps, model_results = my_hyperparameter_tuning(x_train, y_one_hot_train, x_val, y_one_hot_val)
    # print(f"Running hyper-parameter tuning")
    # tuner, best_hps = hyperparameter_tuning(x_train, y_one_hot_train, x_val, y_one_hot_val)
    # print(f"{best_hps=}")
    best_hps = {'num_of_neurons_lstm': 128, 'num_of_neurons_dense': 64, 'l2': 0.0}

    model = Sequential()
    model.add(LSTM(units=best_hps.get('num_of_neurons_lstm'), kernel_regularizer=L2(best_hps.get('l2')),
                   return_sequences=True, input_shape=(None, len(features_to_use))))
    model.add(LSTM(units=best_hps.get('num_of_neurons_lstm'), kernel_regularizer=L2(best_hps.get('l2'))))
    model.add(Dense(units=best_hps.get('num_of_neurons_dense'), kernel_regularizer=L2(best_hps.get('l2')),
                    activation='relu'))
    model.add(Dense(units=2, kernel_regularizer=L2(best_hps.get('l2')), activation='softmax'))
    model.summary()

    print("Training...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score_mine, 'AUC', 'accuracy'])  # loss='mean_squared_error', metrics=['accuracy'])
    hist = model.fit(x_train, y_one_hot_train, epochs=3, batch_size=32, verbose=1, validation_data=(x_val, y_one_hot_val), shuffle=True)
    print(f"{hist.history=}")
    model.save('lstm_model.h5')



def predict(test_directory_path, features_to_use=used_features, is_shap=False):
    model = keras.models.load_model('lstm_model.h5', custom_objects={"f1_score_mine": f1_score_mine})
    ########### Test: #############
    print()
    print("Testing...")
    _, _, test_df_list = read_data(get_filenames(test_directory_path))
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
    pred_df.to_csv('lstm_prediction.csv', index=False, header=False)

    # confusion matrix
    conf_mat = confusion_matrix(y_act, y_preds)
    ax = sns.heatmap(conf_mat, annot=True, cmap='Blues')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    ## Display the visualization of the Confusion Matrix.
    plt.show()

    if is_shap:
        print("Kmeans for shap values...")
        # compute SHAP values
        print(f"Loading lstm data")
        x_train = np.load('lstm_data/x_train.npy')
        # y_train = np.load('lstm_data/y_train.npy')
        # train_pids = np.load('lstm_data/train_pids.npy')
        # data = shap.kmeans(x_test, 100).data
        explainer = shap.DeepExplainer(model, x_train)
        # explainer = shap.DeepExplainer(model, x_train[0].reshape(1, x_train.shape[1], x_train.shape[2]))
        # x_test = x_test[0].reshape(x_test.shape[1], x_test.shape[2])
        shap_values = explainer.shap_values(x_test[0])


        # explainer = shap.KernelExplainer(clf.predict, data)
        # shap_values = explainer.shap_values(data)

        print(f"Shap values length: {len(shap_values)}\n")
        print(f"Sample shap value:\n{shap_values[0]}")
        shap.summary_plot(shap_values, x_test[0], plot_type="bar",
                          feature_names=features_to_use, plot_size=(12, 12), show=False)
        plt.savefig('mlp_shap_pics/bar_plot.png')
        shap.summary_plot(shap_values, feature_names=features_to_use, cmap=plt.get_cmap("winter_r"),
                          plot_size=(10, 12), show=False)
        plt.savefig('mlp_shap_pics/dot_plot.png')

    test_auc = roc_auc_score(y_act, y_pred[:, 1])
    print('test dataset AUC: ' + str(test_auc))
    acc = accuracy_score(y_act, y_preds)
    print('test dataset acc: ' + str(acc))
    f1 = f1_score(y_act, y_preds)
    print("test dataset F1: ", f1)
    pass


def check_predictions(prediction_path, test_directory_path):
    pred_df = pd.read_csv(prediction_path, header=None)  # should be sorted by ids ascending
    ids_pred = pred_df.iloc[:, 0].to_list()
    y_preds = pred_df.iloc[:, 1].to_list()
    filenames = get_filenames(test_directory_path)
    y_true = []
    ids_true = []
    for filename in tqdm(filenames):
        with filename.open() as fp:
            patient_id = str(filename).split("_")[1].split(".")[0]
            # assert ids_pred[i] == int(patient_id)
            ids_true.append(int(patient_id))
            df = pd.read_csv(fp, sep='|')
            if 1 in list(df['SepsisLabel']):
                y_true.append(1)
            else:
                y_true.append(0)
    true_df = pd.DataFrame(data={'Id': ids_true, 'SepsisLabel': y_true})
    true_df.sort_values(by='Id', ascending=True, inplace=True)
    y_true = true_df['SepsisLabel'].to_list()
    f1 = f1_score(y_true, y_preds)
    print(f"F1 score: {f1}")


if __name__ == "__main__":
    # train(make_matrix=True)
    # predict('data/test', is_shap=False)
    # check_predictions('lstm_prediction.csv', 'data/test')
    pass

