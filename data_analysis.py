import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import powerlaw
from pathlib import Path
from tqdm import tqdm


vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp', 'EtCO2']
labs = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
        'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
        'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
demogs = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
label = ['SepsisLabel']
nonlabels = vitals + labs + demogs


def get_filenames(directory_path):
    filepath = Path(directory_path)
    filenames = [fname for fname in filepath.iterdir() if fname.is_file() and fname.suffix == '.psv']
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
            patient_id = str(filename).split("_")[1].split(".")[0]
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


def hist(df, features, file_name):
    for i in range(len(features)):
        sns.distplot(df[features[i]].dropna(), kde=True, bins=30, hist=True, hist_kws={"edgecolor": 'black'})
        plt.title('Histogram of ' + str(features[i]))
        plt.xlabel(features[i])
        plt.ylabel('Density')
        plt.savefig(file_name + str(features[i]) + '_hist.png')
        plt.close('all')


def hist_after_imp(df, df_imp, features, file_name):
    for i in range(len(features)):
        sns.distplot(df[features[i]].dropna(), kde=True, bins=30, hist=False, color='blue', label='Original')
        sns.distplot(df_imp[features[i]], kde=True, bins=30, hist=False, color='red', label='After imputation')
        plt.title('Distribution function of ' + str(features[i]) + ' before and after imputation')
        plt.xlabel(features[i])
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(file_name + str(features[i]) + '_hist.png')
        plt.close('all')


def qqplot(df, features, file_name):
    for i in range(len(features)):
        fig = sm.qqplot(df[features[i]].dropna(), stats.t, line="s", fit=True)
        plt.title('QQ-plot of ' + str(features[i]))
        plt.savefig(file_name + str(features[i]) + '_qqplot.png')
        plt.close('all')
        del fig


def log_log(df, features, file_name):
    for i in range(len(features)):
        fit = powerlaw.Fit(df[features[i]].dropna())
        fit.power_law.plot_pdf(color='r', linestyle='--')
        fit.plot_pdf(color='blue', label=features[i])
        plt.title('Log-log of ' + str(features[i]))
        plt.xlabel('log ' + str(features[i]))
        plt.ylabel('log Count')
        plt.savefig(file_name + str(features[i]) + '_log_log.png')
        plt.close('all')
        del fit


def heatmap(df, features, file_name, size):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool_))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, mask=mask, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                cmap=cmap, fmt='.2f', annot_kws={"size": size})
    plt.suptitle("Heat-map for variables in " + features, fontsize=22, y=0.83)
    plt.savefig(file_name + features + '_scatter.png', dpi=250)
    plt.close('all')


if __name__ == "__main__":
    print("\nTrain set:")
    all_df = read_data(get_filenames('data/train'))
    df_concat = pd.concat(all_df)
    df_concat_imp = impute_missing_vals(df_concat, nonlabels)
    dfs = [df_concat[vitals], df_concat[labs], df_concat[demogs]]
    dfs_imp = [df_concat_imp[vitals], df_concat_imp[labs], df_concat_imp[demogs]]
    features = [vitals, labs, demogs]

    # Plot histograms without nans
    print("\nSaving histograms...")
    save_files = ['histograms/vitals_hist_imgs/', 'histograms/labs_hist_imgs/', 'histograms/demogs_hist_imgs/']
    for i in range(len(dfs)):
        hist(dfs[i], features[i], save_files[i])

    # Plot distributions before and after imp
    print("\nSaving distributions before and after imp...")
    save_files = ['histograms_imp/vitals/', 'histograms_imp/labs/', 'histograms_imp/demogs/']
    for i in range(0, len(dfs)):
        hist_after_imp(dfs[i], dfs_imp[i], features[i], save_files[i])

    # Plot qqplots before and after imp
    print("\nSaving qqplots...")
    save_files = ['qqplots/vitals_qqplot/', 'qqplots/labs_qqplot/', 'qqplots/demogs_qqplot/']
    for i in range(len(dfs)):
        qqplot(dfs[i], features[i], save_files[i])

    # Plot log-log before and after imp
    print("\nSaving log-log plots...")
    save_files = ['log_log/vitals_log_log/', 'log_log/labs_log_log/', 'log_log/demogs_log_log/']
    for i in range(len(dfs)-1):
        log_log(dfs[i], features[i], save_files[i])

    # Heatmap for variables
    print("\nSaving heat-maps...")
    heatmap(df_concat[vitals + demogs + label], 'Vital signs and Demographics', 'heatmaps/', 10)
    heatmap(df_concat[labs + label], 'Laboratory values', 'heatmaps/', 5)

    # Df of nan ratio for each var in labs
    print("Nan ratio for each variable in labs")
    nan_ratio = [np.mean((df_concat[var].isnull().sum() / df_concat[var].shape[0])) for var in labs]
    df = pd.DataFrame(data={'variable': labs, 'nan_ratio': nan_ratio})
    df.sort_values(by=['nan_ratio'], inplace=True, ascending=False, ignore_index=True)
    print(df)

    # Df of p values (normal tests) for each var
    print("\nP values for each variable")
    p_values_normaltest = [stats.normaltest(df_concat_imp[var], nan_policy='omit')[1] for var in nonlabels]
    p_values_shapiro = [stats.shapiro(df_concat_imp[var].dropna())[1] for var in nonlabels]
    p_values_jarque_bera = [stats.jarque_bera(df_concat_imp[var].dropna())[1] for var in nonlabels]
    df = pd.DataFrame(data={'variable': nonlabels, 'normaltest': p_values_normaltest,
                            'shapiro': p_values_shapiro, 'jarque_bera': p_values_jarque_bera})
    df.sort_values(by=['jarque_bera'], inplace=True, ascending=False, ignore_index=True)
    print(df)
