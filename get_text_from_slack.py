'''
週末報告から、テキストを取り出し。
参考: https://qiita.com/yoshii0110/items/2a7ea29ca8a40a9e42f4
'''


import requests
from datetime import datetime
import json
import time
from tqdm import tqdm
import csv

SLACK_CHANNEL_ID = '******'
TOKEN = "******"

SLACK_URL = "https://slack.com/api/conversations.history"
REPLY_URL = "https://slack.com/api/conversations.replies"
oldest_unix = int(datetime(2020,12,1,0,0).timestamp())
latest_unix = int(datetime(2022,3,10,0,0).timestamp())

headersAuth = {
    'Authorization': 'Bearer '+ str(TOKEN),
    }  

def get_replies(msg):
    time.sleep(1.5)
    payload = {
        "channel": SLACK_CHANNEL_ID,
        "ts": msg['ts'],
        "limit": 100
        }
    response = requests.get(REPLY_URL, headers=headersAuth, params=payload)
    json_data = response.json()
    if json_data:
        return json_data['messages']
    else:
        return None


def main():
    f = open("./data/user_data/weekly_message_20220313.csv", 'w', newline='')
    data = ['client_msg_id','type','text']
    writer = csv.writer(f)
    writer.writerow(data)

    oldest = oldest_unix
    while oldest < latest_unix:
        payload = {
            "channel": SLACK_CHANNEL_ID,
            "oldest": str(oldest),
            "limit": 100
            }
        response = requests.get(SLACK_URL, headers=headersAuth, params=payload)
        json_data = response.json()
        msgs = json_data['messages']

        for msg in tqdm(msgs):
            if 'client_msg_id' not in msg:
                continue
            replies = get_replies(msg)
            if replies:
                for reply in replies:
                    if not reply['text'].startswith('```'):
                        row = [reply['client_msg_id'], reply['type'], reply['text']]
                        writer.writerow(row)
        
        oldest = float(msgs[0]['ts'])

    f.close()
    return print("Suceess!")

if __name__ == "__main__":
    main()