import warnings
warnings.simplefilter("ignore", UserWarning)

from sklearn import metrics
import pandas as pd
import argparse


def load_data(csv):
    d = pd.read_csv(csv, index_col=0)
    return d


def clean_open_ai_data(df):
    """This function is for cleaning OpenAI's label output that can sometimes get garbled. Add lines as you see new
    weirdness appear """
    df['predicted_label'] = df.predicted_label.str.replace('\n', '')
    df['predicted_label'] = df.predicted_label.str.replace('Practice: ', '')
    #df['predicted_label'] = df['predicted_label'].str.replace(r'[^\w\s]', '')
    df.loc[df.predicted_label == 'Community Work', 'predicted_label'] = 'Community work'
    df.loc[df.predicted_label == 'Shit', 'predicted_label'] = 'Shitposting'
    df.loc[df.predicted_label == 'Community', 'predicted_label'] = 'Community work'
    df.loc[df.predicted_label == 'Boost', 'predicted_label'] = 'Boosting'
    df.loc[df.predicted_label == 'Advoc', 'predicted_label'] = 'Advocacy'
    df.loc[df.predicted_label == 'Express', 'predicted_label'] = 'Expressing solidarity'
    df.loc[df.predicted_label == 'L1. Tweets that request ', 'predicted_label'] = 'Membership requests'
    df.loc[df.predicted_label == '\n\nNot applicable', 'predicted_label'] = 'Not applicable'
    df['predicted_label'] = df.predicted_label.str.strip()
    df['true_label'] = df.true_label.apply(lambda x: x[5:])
    df.loc[~df.predicted_label.isin(df.true_label.to_list()), 'predicted_label'] = 'Not applicable'
    return df


def score_open_ai_data(data):
    actual = data.true_label.to_list()
    predicted = data.predicted_label.to_list()
    print(metrics.classification_report(actual, predicted, zero_division=0))
    report = metrics.classification_report(actual, predicted, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def main():
    ap = argparse.ArgumentParser(description='A script to clean and evaluate OpenAI output')
    ap.add_argument('-i', '--input_file', required=True, help='Path to file with OpenAI predicted data and true labels')
    ap.add_argument('-r', '--result_file', required=True, help='Path of the result (output) file')
    args = ap.parse_args()
    dataset = load_data(args.input_file)
    clean_dataset = clean_open_ai_data(dataset)
    clean_dataset.to_csv('cleaned_test.csv')
    scores = score_open_ai_data(clean_dataset)
    scores.to_csv(f'{args.result_file}')


if __name__ == '__main__':
    main()
