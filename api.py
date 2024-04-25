import requests
import json
from tqdm import tqdm
import os
import time
import argparse

URLS=json.load(open('urls.json','r'))

def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=N8ZLDrC7ScEXgFYW3IlJdQh2&client_secret=XzSELlx7gHCe0h6jiOrOPKIsn67FCyAJ"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def get_url(model):
    return URLS[model]

def api_use(args):
    url = get_url(args.model)
    return API(url=url)


class API:
    def __init__(self, url=None):
        self.url = url
    def get_response(self, prompts):

        url = self.url
        url = f"{url}?access_token=" + get_access_token()

        responses = []

        for prompt in prompts:

            # 尝试三次请求，如果失败则跳过
            while True:
                try:
                    payload = json.dumps({
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Content-Type': 'application/json'
                    }

                    response = requests.request("POST", url, headers=headers, data=payload).json()
                    responses.append(response['result'])
                    break
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    continue
        print(responses)
        return responses