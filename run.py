import glob
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import json

import torch
from rdt.transformers import GaussianNormalizer, LabelEncoder


from model.uniformgan import UniformGAN
from model.eval.evaluation import get_utility_metrics, privacy_metrics, stat_sim

data_set_names = ['Adult', 'Covtype', '', 'Insurance','Intrusion']
replication_n = 1
cache_path = 'pickles/'
epochs = 50
version = f'-{epochs}-epochs-log-mixed-categorical'
fake_root = 'fake_datasets/'


def generate(data_sets):
    for data_set in data_sets:
        raw_csv_path = f'datasets/numeric/{data_set}.csv'
        categorical_columns, log_columns, mixed_columns, meta_data, sdtypes, transformers = config_from_metadata(data_set)

        synthesizer = UniformGAN(
            raw_csv_path=raw_csv_path,
            sdtypes=sdtypes,
            transformers=transformers,
            log_columns=log_columns,
            mixed_columns=mixed_columns,
            categorical_columns=categorical_columns,
            problem_type={meta_data['problem_type']: meta_data['tables'][data_set.lower()]['target']}
        )
        synthesizer.get_config()
        synthesizer.synthesizer.epochs = epochs
        synthesizer.fit()

        for i in range(replication_n):
            fake_path = fake_root + data_set + version + str(i) + ".csv"
            fake_data = synthesizer.generate_samples()
            fake_data.to_csv(fake_path, index=False)


def config_from_metadata(data_set):
    meta_data_path = f'datasets/metadata/{data_set}.json'
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)
    sdtypes = {}
    for k, v in meta_data['tables'][data_set.lower()]['fields'].items():
        sdtypes.update({k: v['type']})
    transformers = {}
    mixed_columns = {}
    log_columns = []
    categorical_columns = []
    for k, v in meta_data['tables'][data_set.lower()]['fields'].items():
        if v['type'] == 'numerical':
            if v['subtype'] == 'integer':
                # if the subtype is integer we learn the rounding scheme
                transformers.update({k: GaussianNormalizer(
                                                           enforce_min_max_values=True,
                                                           learn_rounding_scheme=True,
                                                           model_missing_values=True)})
            if v['subtype'] == 'float':
                transformers.update({k: GaussianNormalizer(
                                                           enforce_min_max_values=True,
                                                           model_missing_values=True)})
            if v.get('subtype') == 'None':
                transformers.update({k: GaussianNormalizer(
                                                           enforce_min_max_values=True,
                                                           learn_rounding_scheme=True,
                                                           model_missing_values=True)})
            if v.get('d_hint') == 'log':
                log_columns.append(k)
            if v.get('d_hint') == 'mixed':
                mixed_columns.update({k: [0.0]})
        if v['type'] == 'categorical':
            transformers.update({k: LabelEncoder()})
            categorical_columns.append(k)
    return categorical_columns, log_columns, mixed_columns, meta_data, sdtypes, transformers


def analyse_model(data_sets):
    fake_paths = {}
    for dataset_name in data_sets:
        for i in range(replication_n):
            synth_path = fake_root + dataset_name + version + str(i) + ".csv"
            fake_paths.setdefault(dataset_name, []).append(synth_path)
        paths = fake_paths[dataset_name]
        # Specifying the categorical columns of the dataset used
        real_path = f"datasets/numeric/{dataset_name}.csv"
        categorical_columns, log_columns, mixed_columns, meta_data, sdtypes, transformers = config_from_metadata(dataset_name)


        # utility for classifier target
        if meta_data['problem_type'] == 'Regression':
            # Specifying the list of classifiers to conduct ML utility evaluation
            model_dict = {"Regression": ["l_reg", "ridge", "lasso", "B_ridge"]}
            result_mat = get_utility_metrics(real_path, paths, "MinMax", model_dict, test_ratio=0.20)

            result_df = pd.DataFrame(result_mat,
                                     columns=["Mean_Absolute_Percentage_Error", "Explained_Variance_Score", "R2_Score"])
            result_df.index = list(model_dict.values())[0]
            result_df.to_csv(f'results/{dataset_name}{version}utility_metrics.csv')
            print(result_df.to_string())

        # utility for regression target
        if meta_data['problem_type'] == 'Classification':
            # Specifying the list of classifiers to conduct ML utility evaluation
            model_dict = {"Classification": ["lr", "dt", "rf", "mlp", "svm"]}
            # Storing and presenting the results as a dataframe
            result_mat = get_utility_metrics(real_path, paths, "MinMax", model_dict, test_ratio=0.20)
            result_df = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
            result_df.index = model_dict['Classification']
            result_df.to_csv(f'results/{dataset_name}{version}utility_metrics.csv')
            print(result_df.to_string())

        stat_res_avg = []
        for fake_path in paths:
            stat_res = stat_sim(real_path, fake_path, categorical_columns)
            stat_res_avg.append(stat_res)

        # Storing and presenting the results as a dataframe
        stat_columns = ["Average WD (Continuous Columns", "Average JSD (Categorical Columns)", "Correlation Distance"]
        stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns)
        stat_results.to_csv(f'results/{dataset_name}{version}stat_results.csv')
        print(stat_results.to_string())

        priv_res_avg = []
        for fake_path in paths:
            priv_res = privacy_metrics(real_path, fake_path)
            priv_res_avg.append(priv_res)

        privacy_columns = ["DCR between Real and Fake (5th perc)", "DCR within Real(5th perc)",
                           "DCR within Fake (5th perc)", "NNDR between Real and Fake (5th perc)",
                           "NNDR within Real (5th perc)", "NNDR within Fake (5th perc)"]
        privacy_results = pd.DataFrame(np.array(priv_res_avg).mean(axis=0).reshape(1, 6), columns=privacy_columns)
        privacy_results.to_csv(f'results/{dataset_name}{version}privacy_results.csv')
        print(privacy_results.to_string())


generate(data_set_names)
analyse_model(data_set_names)
