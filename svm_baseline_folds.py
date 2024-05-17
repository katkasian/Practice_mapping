import argparse
import winsound
import os

import warnings
warnings.simplefilter('ignore', UserWarning)
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics


def load_data(folder, keyword):
    trains = []
    tests = []
    for n, i in enumerate(list(os.listdir(folder))):
        if keyword in i and n % 2 == 1:
            d = pd.read_csv(f'{folder}/{i}', index_col=0)
            d.columns = ['data', 'target']
            tests.append(d)
        elif keyword in i and n % 2 == 0:
            d = pd.read_csv(f'{folder}/{i}', index_col=0)
            d.columns = ['data', 'target']
            trains.append(d)
    train_test_tuples = list(zip(tests, trains))
    return train_test_tuples


def run_linear_svm(train_test):
    feature_extraction = TfidfVectorizer()
    X = feature_extraction.fit_transform(np.append(train_test[0].data.values, train_test[1].data.values))
    docs_train = X[:train_test[0].shape[0]]
    docs_test = X[train_test[0].shape[0]:]
    y_train = train_test[0].target
    y_test = train_test[1].target
    clf = SVC(kernel='linear')
    clf.fit(docs_train, y_train)
    y_predicted = clf.predict(docs_test)
    report = metrics.classification_report(y_test, y_predicted, output_dict=True, zero_division=0)
    output_df = pd.DataFrame(report).transpose()
    return output_df


def run_weighted_svm(train_test):
    feature_extraction = TfidfVectorizer()
    X = feature_extraction.fit_transform(np.append(train_test[0].data.values, train_test[1].data.values))
    docs_train = X[:train_test[0].shape[0]]
    docs_test = X[train_test[0].shape[0]:]
    y_train = train_test[0].target
    y_test = train_test[1].target
    clf = SVC(class_weight='balanced')
    clf.fit(docs_train, y_train)
    y_predicted = clf.predict(docs_test)
    report = metrics.classification_report(y_test, y_predicted, output_dict=True, zero_division=0)
    output_df = pd.DataFrame(report).transpose()
    return output_df


def calculate_scores_across_folds(result_df, output_dir, output_name):
    by_row_index = result_df.groupby(result_df.index)
    df_means = by_row_index.mean()
    df_std = by_row_index.std()
    print(df_means)
    print(df_std)
    df_means.to_csv(f'{output_dir}\\{output_name}_mean.csv')
    df_std.to_csv(f'{output_dir}\\{output_name}_std.csv')


def main():
    ap = argparse.ArgumentParser(description='A script to create an SVM baseline for a dataset.')
    ap.add_argument('-i', '--input_folder', required=True,
                    help='Path to folder from which to calculate an SVM baseline')
    ap.add_argument('-c', '--case_study', required=True, help='Case study: NAFO or ESC')
    ap.add_argument('-r', '--result_file', required=True, help='Name of the result (output) file')
    ap.add_argument('-d', '--result_folder', required=True, help='Path of the result (output) file')
    ap.add_argument('-v', '--version', required=True, choices={'l', 'w'}, help='Choose linear or weighted SVM')
    args = ap.parse_args()
    folds = load_data(args.input_folder, args.case_study)
    results = []
    for fold in folds:
        if args.version == 'l':
            report = run_linear_svm(fold)
        else:
            report = run_weighted_svm(fold)
        results.append(report)
    print(results)
    df_results = pd.concat(results)
    calculate_scores_across_folds(df_results, args.result_folder, args.result_file)
    duration = 1000  # milliseconds
    freq = 2000  # Hz
    winsound.Beep(freq, duration)


if __name__ == '__main__':
    main()
