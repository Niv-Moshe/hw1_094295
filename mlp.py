from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import numpy as np
from hyperopt import STATUS_OK, hp, fmin, tpe
from tqdm import tqdm
import pickle
import random
import shap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

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
    filenames = [fname for fname in filepath.iterdir() if fname.is_file() and fname.suffix == '.psv']  # [:1000]
    all_ids = []

    for filename in filenames:
        patient_id = str(filename).split("_")[1].split(".")[0]
        all_ids.append(int(patient_id))

    filenames_df = pd.DataFrame(data={'Id': all_ids, 'filename': filenames})
    filenames_df.sort_values(by='Id', ascending=True, inplace=True)
    return filenames_df['filename'].to_list()


def read_data(filenames):
    all_df = []
    sick, healthy = 0, 0
    for filename in tqdm(filenames):
        with filename.open() as fp:
            patient_id = str(filename).split("_")[-1].split(".")[0]
            df = pd.read_csv(fp, sep='|')
            df['pid'] = len(df) * [int(patient_id)]
            if 1 in list(df['SepsisLabel']):
                ind = list(df['SepsisLabel']).index(1)
                df = df[:ind + 1]
                sick += 1
            else:
                healthy += 1
            all_df.append(df)
    print("Got", len(all_df), "samples:", sick, "sick and", healthy, "healthy")
    return all_df


def mean_df(df_list):
    patients_mean_list = []
    pids = []
    for patient_df in tqdm(df_list):
        pids.append(int(patient_df['pid'].to_list()[0]))
        # patient_df.drop(['pid'], axis=1, inplace=True)
        patient_df_clean = impute_missing_vals(patient_df, nonlabels)
        label = 1 if 1 in patient_df_clean[labels].to_numpy() else 0
        patient_df_mean = patient_df_clean.mean()
        patient_df_mean[labels] = label
        patients_mean_list.append(patient_df_mean)
    data = pd.concat(patients_mean_list, axis=1, ignore_index=True).T
    # print(1 in data[labels].to_numpy()[:, 0])
    data[labels] = data[labels].apply(np.int64)
    # print(1 in data[labels].to_numpy()[:, 0])
    print("Finished filling missing values")
    return data, pids


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
    # print("Finished filling missing values")
    return df_clean


def BO_TPE_mlp(train, val):
    print("Hyper parameter tuning...")

    def objective(params):
        # clf = MLPClassifier(params,
        #                     random_state=42, max_iter=300, verbose=False)
        clf = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'],
                            batch_size=params['batch_size'],
                            alpha=params['alpha'],
                            learning_rate_init=params['learning_rate_init'],
                            early_stopping=params['early_stopping'],
                            max_iter=300, verbose=False, random_state=42)
        clf.fit(train[nonlabels].values, train[labels].values.ravel())

        y_val_class = clf.predict(val[nonlabels].values)
        f1 = f1_score(val[labels].to_numpy(), np.array(y_val_class))
        loss = 1 - f1
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    early_stoppings = [True, False]
    hidden_layer_sizes = [(1000, 1000, 1000), (100, 100), (100, 100, 100), (100, 100, 100, 100)]
    batch_sizes = [32, 64, 128]
    learning_rates = [0.01, 0.001, 1e-4, 0.1]
    alphas = [0.0, 0.0001, 0.001, 0.01]

    space = {'early_stopping': hp.choice('early_stopping', early_stoppings),
             'hidden_layer_sizes': hp.choice('hidden_layer_sizes', hidden_layer_sizes),
             'batch_size': hp.choice('batch_size', batch_sizes),
             'learning_rate_init': hp.choice('learning_rate_init', learning_rates),
             'alpha': hp.choice('alpha', alphas)}

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)

    best_param = {'early_stopping': early_stoppings[(best['early_stopping'])],
                  'hidden_layer_sizes': hidden_layer_sizes[(best['hidden_layer_sizes'])],
                  'batch_size': batch_sizes[(best['batch_size'])],
                  'learning_rate_init': learning_rates[(best['learning_rate_init'])],
                  'alpha': alphas[(best['alpha'])]}

    return best_param


def train(features_to_use=used_features):
    ########### Train: #############
    print("\nTrain set:")

    train_df_list = read_data(get_filenames('data/train'))
    all_train, _ = mean_df(train_df_list)
    # val = all_train.sample(frac=0.1, random_state=42)
    # train_df = all_train.drop(val.index)
    train_df = all_train

    ########### Model: #############
    print("\nMLP:")
    # best_param = BO_TPE_mlp(train_df, val)
    # best_param = {'early_stopping': True, 'hidden_layer_sizes': (100, 100, 100, 100, 100), 'batch_size': 128,
    #               'learning_rate_init': 0.0001, 'alpha': 0.1}
    # print(f"{best_param=}")
    print()
    print("Training...")
    # clf = MLPClassifier(hidden_layer_sizes=best_param['hidden_layer_sizes'],
    #                     batch_size=best_param['batch_size'],
    #                     alpha=best_param['alpha'],
    #                     learning_rate_init=best_param['learning_rate_init'],
    #                     early_stopping=best_param['early_stopping'],
    #                     max_iter=300, verbose=True, random_state=42)
    # clf.fit(train_df[features_to_use].values, train_df[labels].values.ravel())
    clf = MLPClassifier(batch_size=64, max_iter=300, random_state=42, verbose=True, early_stopping=True)
    clf.fit(train_df[features_to_use].values, train_df[labels].values.ravel())
    # save
    with open('mlp_model.pkl', 'wb') as f:
        pickle.dump(clf, f)


def predict(test_directory_path, features_to_use=used_features, is_shap=False):
    # load
    with open('mlp_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    ########### Test: #############
    print("\nTest set:")
    test_df_list = read_data(get_filenames(test_directory_path))  # filenames come in messy order
    test, test_pids = mean_df(test_df_list)
    # test.to_pickle('test.pkl')

    y_pred = clf.predict(test[features_to_use].values)
    y_pred_prob = clf.predict_proba(test[features_to_use].values)
    test_auc = roc_auc_score(test[labels], y_pred_prob[:, 1])
    print('test dataset AUC: ' + str(test_auc))
    acc = accuracy_score(test[labels], y_pred)
    print('test dataset acc: ' + str(acc))
    f1 = f1_score(test[labels], y_pred)
    print("test dataset F1:", f1)

    y_pred_int = [int(y) for y in y_pred]
    pred_df = pd.DataFrame(data={'Id': test_pids, 'SepsisLabel': y_pred_int})
    pred_df.sort_values(by=['Id'], ascending=True, inplace=True)
    pred_df.to_csv('mlp_prediction.csv', index=False, header=False)

    # confusion matrix
    conf_mat = confusion_matrix(test[labels], y_pred_int, [0, 1])
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
        data = shap.kmeans(test[features_to_use].values, 100).data
        explainer = shap.KernelExplainer(clf.predict, data)
        shap_values = explainer.shap_values(data)

        print(f"Shap values length: {len(shap_values)}\n")
        print(f"Sample shap value:\n{shap_values[0]}")
        shap.summary_plot(shap_values, data, plot_type="bar",
                          feature_names=features_to_use, plot_size=(12, 12), show=False)
        plt.savefig('mlp_shap_pics/bar_plot.png')
        shap.summary_plot(shap_values, feature_names=features_to_use, cmap=plt.get_cmap("winter_r"),
                          plot_size=(10, 12), show=False)
        plt.savefig('mlp_shap_pics/dot_plot.png')
    pass


def check_predictions(prediction_path, test_directory_path):
    pred_df = pd.read_csv(prediction_path, header=None)  # should be sorted by ids ascending
    ids_pred = pred_df.iloc[:, 0].to_list()
    y_preds = pred_df.iloc[:, 1].to_list()
    filenames = get_filenames(test_directory_path)  # messy file names
    y_true = []
    ids_true = []
    # messy true ids
    for filename in tqdm(filenames):
        with filename.open() as fp:
            patient_id = str(filename).split("_")[-1].split(".")[0]
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
    # train()
    # predict('data/test', is_shap=True)
    # check_predictions('mlp_prediction.csv', 'data/test')
    pass
