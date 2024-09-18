import requests as rq
import pandas as pd
import time

url = 'http://127.0.0.1:8000/batch_predict/'
header = {"Content-Type": "application/json; charset=utf-8"}

df = pd.read_csv('assets/' + 'subsampled_test_df_lightCNN_timeClassifier.csv', index_col = 0)

number = 50 # change number to the desired sample space
sub_df = df.sample(number)
data = []
for index, row in sub_df.iterrows():
    data.append({'record': row.iloc[:2048].tolist()})

st = time.time()

collected_data = rq.post(url = url, headers = header, json = data).json()

et = time.time()

print('\n', 'All the predictions are correct: ', sub_df['state'].to_list() == [item['Prediction'] for item in collected_data["Analysis"]])

print('\n', 'Execution Time (mSec): ', round(collected_data['Execution Time'], 4), ' - ', 'Request Handling Time (mSec): ', round(1000 * (et - st), 4))
