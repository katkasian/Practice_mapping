import sys
import warnings
warnings.simplefilter("ignore", UserWarning)


import os
import pandas as pd
import re
import argparse
import string


from datetime import datetime
from nltk.tokenize import word_tokenize


from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from textwrap import dedent
from openai import ContentFilterFinishReasonError

# make sure your BQ credentials are stored in your .env file under variable name 'GOOGLE_APPLICATION_CREDENTIALS'
from google.cloud import bigquery
from google.cloud.bigquery.client import Client


GBQ_PROJECT = 'dmrc-data'
DS = 'ua_nafo_main'  # change the name of the dataset here
TB = 'tweets'
LIM = 10000  # change the number of the items needed
CLST_TB = 'degree5_mod_1_clusters'  # change the name of modularity table here
CLST_EXCL = ['8']

load_dotenv()
CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = 'gpt-4o'


class LabeledTweet(BaseModel):
    label: str
    text: str


class LabeledTweets(BaseModel):
    tweets: list[LabeledTweet]


def load_data_from_bq(dataset, table, cluster_table, clusters_to_exclude, limit):
    bq = Client(project=GBQ_PROJECT)
    query_string = f"""
WITH already_labeled AS (
  SELECT tweet_id 
  FROM ua_nafo_main.gpt_labeled_practices
  GROUP BY tweet_id  -- This ensures we only get one row per tweet_id regardless of duplicates
),
eligible_tweets AS (
  SELECT DISTINCT twt.tweet_id, twt.tweet_text 
  FROM {dataset}.{table} twt 
  LEFT JOIN {dataset}.{cluster_table} clust 
      ON twt.author_id = clust.Id 
  LEFT JOIN {dataset}.{cluster_table} ref_clust 
      ON twt.referenced_tweet_author_id = ref_clust.Id 
  LEFT JOIN already_labeled al 
      ON twt.tweet_id = al.tweet_id
  WHERE twt.reference_level = '0'
    AND clust.modularity_class NOT IN UNNEST({clusters_to_exclude})
    AND (ref_clust.modularity_class IS NULL 
         OR ref_clust.modularity_class NOT IN UNNEST({clusters_to_exclude}))
    AND tweet_type != 'retweet'
    AND NOT contains_substr(tweet_text, 'MAKS_NAFO_FELLA')
    AND author_username != 'nafo_article_5b'
    AND al.tweet_id IS NULL
)
SELECT * FROM eligible_tweets
LIMIT {limit}
    """
    print(f"Querying '{table}' table...")
    df = (bq.query(query_string).result().to_dataframe())
    return df


def clean_data_from_bq(data):
    data['clean_text'] = data.tweet_text.apply(lambda x: re.sub(r'(#\w*?)\s|&', '', x))
    data['clean_text'] = data.clean_text.apply(lambda x: re.sub(r'(@\w*?)\s|&', '', x))
    data['clean_text'] = data.clean_text.apply(lambda x: re.sub(r'http\S+', '', x))
    data['clean_tokenized'] = data.clean_text.apply(lambda x: word_tokenize(x))
    data['n_tokens'] = data.clean_tokenized.apply(lambda x: len(x))
    data = data.loc[data.n_tokens > 3].copy(deep=True)
    data.drop(columns=['clean_tokenized', 'n_tokens', 'clean_text'], inplace=True)
    print('Cleaned BQ data')
    if data.shape[0] == 0:
        print('... But got nothing left to send to OpenAI')
        sys.exit()
    print(f'Left with {data.shape[0]} texts')
    return data.head(15) #making sure prompt is not too long


def prepare_openai_prompt(twt_data):
    with open('mar_2024_prompts/v6_cot/NAFO_MPE_COT_0_with_system_instruction.txt', encoding='utf-8') as f:
        cot_mpe_prompt = f.read()
        tweets = twt_data.tweet_text.to_list()
        content = ''
    for n, i in enumerate(tweets):
        content += f'Tweet_{n}:{i}\n'
    print('Prepared OpenAI prompt')
    return content, cot_mpe_prompt


def get_tweet_label(twt_text, prompt):
    try:
        completion = CLIENT.beta.chat.completions.parse(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": dedent(prompt)},  # prompt goes here
                {"role": "user", "content": twt_text}],
            response_format=LabeledTweets,
            max_completion_tokens=5000
        )
        print('Received response from OpenAI')
        return completion.choices[0].message.parsed
    except ContentFilterFinishReasonError:
        print('Request did not pass content filter')


def process_open_ai_response(twt_text, prompt):
    results = get_tweet_label(twt_text, prompt)
    if results:
        tweets = results.tweets
        labels = [i.label for i in tweets]
        texts = [i.text for i in tweets]
        result_df = pd.DataFrame([texts, labels]).T
        result_df.columns = ['tweet_text', 'label']
        print('Processed OpenAI response')
        return result_df
    else:
        print(twt_text)
        tweets_could_not_label = twt_text.split('\nTweet_')
        print(tweets_could_not_label)
        cleaner = []
        for i in tweets_could_not_label:
            if i.startswith('Tweet_'):
                to_append = i
            else:
                to_append = 'Tweet_' + i
            print(to_append)
            cleaner.append(to_append)
        failed_df = pd.DataFrame(cleaner)
        failed_df['label'] = 'Potential_content_filter_fail'
        failed_df.columns = ['tweet_text', 'label']
        print('Failed to pass OpenAI content filter but processed response')
        return failed_df


def clean_open_ai_response(response_df, clean_df):
    response_df['tweet_text_final'] = response_df.tweet_text.str.replace(r'Tweet_\d+\:', '', regex=True)
    response_df.columns = ['index_tweet_text', 'label', 'tweet_text']
    response_df['tweet_text'] = response_df.tweet_text.str.replace('\u202F', ' ')
    response_df['tweet_text'] = response_df.tweet_text.str.replace('&amp;', '&')
    response_df['tweet_text'] = response_df.tweet_text.str.replace(r'\W', '', regex=True)
    #response_df['tweet_text'] = response_df.tweet_text.str.replace('\n', '')
    #response_df['tweet_text'] = response_df.tweet_text.str.replace(' ', '')
    #for i in string.whitespace:
    #    response_df['tweet_text'] = response_df.tweet_text.str.replace(f'{i}', '')
    response_df['tweet_text'] = response_df.tweet_text.str.strip()
    clean_df['cleaned_text'] = clean_df.tweet_text.str.replace('\u202F', ' ')
    clean_df['cleaned_text'] = clean_df.cleaned_text.str.replace('&amp;', '&')
    clean_df['cleaned_text'] = clean_df.cleaned_text.str.replace(r'\W', '', regex=True)
    #for i in string.whitespace:
    #    clean_df['cleaned_text'] = clean_df.cleaned_text.str.replace(f'{i}', '')
    #clean_df['cleaned_text'] = clean_df.cleaned_text.str.replace('\n', '')
    #clean_df['cleaned_text'] = clean_df['cleaned_text'] = clean_df.cleaned_text.str.replace(' ', '')
    clean_df['cleaned_text'] = clean_df.cleaned_text.str.strip()
    clean_df['old_text'] = clean_df.tweet_text
    clean_df['tweet_text'] = clean_df.cleaned_text
    labeled_df = clean_df.merge(response_df, on='tweet_text')
    ind_count = 0 #for validation of text matching, remove later
    for i in response_df.iterrows():
        response_tweet = i[1]['tweet_text']
        sent_tweet = clean_df.iloc[ind_count, 1]
        if response_tweet != sent_tweet:
            print('Sent:')
            print(sent_tweet)
            print('Received:')
            print(response_tweet)
        ind_count +=1
        print('---')
    print(response_df.shape[0])
    print('Cleaned OpenAI response')
    return labeled_df


def save_open_ai_response(merged_df):
    now = datetime.now()
    merged_df.to_csv(f'full_datasets_labelled_backup/nafo/labeled_{DS}_{now.strftime("%Y_%m_%d_%H_%M_%S")}.csv')
    print('Saved OpenAI response locally')


def push_labeled_data_to_bq(merged_labeled_df):
    to_push = merged_labeled_df[['tweet_id', 'label']]
    to_push.columns = ['tweet_id', 'practice']
    client = bigquery.Client()
    table_id = f'{DS}.gpt_labeled_practices'
    job_config = bigquery.LoadJobConfig(schema=[
        bigquery.SchemaField('tweet_id', 'STRING'),
        bigquery.SchemaField('practice', 'STRING'),
    ])
    job_config.create_disposition = "CREATE_IF_NEEDED"
    job = client.load_table_from_dataframe(to_push, table_id, job_config=job_config)
    print(f'Push to BQ successful, pushed {to_push.shape[0]} tweets')


def main():
    ap = argparse.ArgumentParser(description='A script to communicate with BigQuery and OpenAI API to label tweets '
                                             'with practice labels')
    ap.add_argument('-n', '--num_iterations', required=True, help='Number of iterations')
    args = ap.parse_args()
    for i in range(0, int(args.num_iterations)):
        print(f'ITERATION {i + 1} of {args.num_iterations}')
        data_df = load_data_from_bq(DS, TB, CLST_TB, CLST_EXCL, LIM)
        clean_data = clean_data_from_bq(data_df)
        content_for_call = prepare_openai_prompt(clean_data)
        processed_results = process_open_ai_response(content_for_call[0], content_for_call[1])

        cleaned_results = clean_open_ai_response(processed_results, clean_data)
        save_open_ai_response(cleaned_results)
        push_labeled_data_to_bq(cleaned_results)
    print('ALL DONE...')


if __name__ == '__main__':
    main()
