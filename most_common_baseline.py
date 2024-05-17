import warnings

warnings.simplefilter("ignore", UserWarning)

from sklearn.dummy import DummyClassifier
from sklearn import metrics
import pandas as pd
import argparse
import winsound



def read_from_csv(file_path):
    try:
        # Open the CSV file for reading
        df = pd.read_csv(file_path, index_col=0)
        return df
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def random_run(data_frame):
    dataset = data_frame.copy(deep=True)
    dataset.columns = ['data', 'target']
    actual = dataset.target.to_list()
    results = []
    for i in range(0, 2):
        dummy_clf = DummyClassifier(strategy='most_frequent')
        dummy_clf.fit(dataset.data, dataset.target)
        dataset['random_score'] = dummy_clf.predict(dataset.data)
        predicted = list(dummy_clf.predict(dataset.data))
        report = metrics.classification_report(actual, predicted, zero_division=0, output_dict=True)
        output_df = pd.DataFrame(report).transpose()
        results.append(output_df)
        df_results = pd.concat(results)
    return df_results


def output_results(result_df, output_dir, output_name):
    by_row_index = result_df.groupby(result_df.index)
    df_means = by_row_index.mean()
    df_std = by_row_index.std()
    print(df_means)
    print(df_std)
    df_means.to_csv(f'{output_dir}\\{output_name}_mean.csv')
    df_std.to_csv(f'{output_dir}\\{output_name}_std.csv')

def main():
    ap = argparse.ArgumentParser(description='A script to create a most common baseline for a dataset.')
    ap.add_argument('-i', '--input_file', required=True, help='Path to file from which to construct a most common baseline')
    ap.add_argument('-r', '--result_file', required=True, help='Name of the result (output) file')
    ap.add_argument('-d', '--result_folder', required=True, help='Path of the result (output) file')
    args = ap.parse_args()
    base_data = read_from_csv(args.input_file)
    random_classification = random_run(base_data)
    output_results(random_classification, args.result_folder, args.result_file)
    duration = 1000  # milliseconds
    freq = 2000  # Hz
    winsound.Beep(freq, duration)


if __name__ == '__main__':
    main()