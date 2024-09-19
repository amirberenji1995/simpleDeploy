import requests as rq
import pandas as pd
import time

render_url = 'https://simpledeploy-erxy.onrender.com/batch_predict/'
local_url = 'http://127.0.0.1:8000/batch_predict/'
header = {"Content-Type": "application/json; charset=utf-8"}

df = pd.read_csv('assets/' + 'subsampled_test_df_lightCNN_timeClassifier.csv', index_col = 0)

exe_time = {}
delivery_time = {}

for i in [1, 5, 10]:
    sub_df = df.sample(i)
    data = []
    for index, row in sub_df.iterrows():
        data.append({'record': row.iloc[:2048].tolist()})
    
    exe_time[i] = {}
    delivery_time[i] = {}
    
    for j in range(10):
        st = time.time()
        collected_data = rq.post(url = render_url, headers = header, json = data, ).json()
        et = time.time()
        
        exe_time[i][j] = collected_data['Execution Time']
        delivery_time[i][j] = 1000 * (et - st)

        print('\n', 'Sample Lenght: ', i, ' - ', 'Iteration: ', j, ' - ', 'Correctness: ', sub_df['state'].to_list() == [item['Prediction'] for item in collected_data["Analysis"]])
        print('\n', 'Execution Time (mSec): ', round(collected_data['Execution Time'], 4), ' - ', 'Request Handling Time (mSec): ', round(1000 * (et - st), 4))

exe_time = pd.DataFrame(exe_time)
print('\n', exe_time, '\n')
print('\n', pd.DataFrame(exe_time).mean(), '\n')

delivery_time = pd.DataFrame(delivery_time)
print('\n', delivery_time, '\n')
print('\n', pd.DataFrame(delivery_time).mean(), '\n')