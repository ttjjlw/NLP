import requests

url='https://www.douyin.com/video/7179199232358763836'

headers = {
            # 无论是否有登陆 都有 cookie
            'cookie': 'douyin.com; ttwid=1%7CHj5s1edW817SdFa5U5hnifjpPs-xPs4Lv7vC7DQNm60%7C1652797030%7C21e300be59bef00949a7d8f790beeeb4db77670205394cc7cce9dcd0229b5e95; odin_tt=fded742fd6268e282962a1c63dd4f62e37e3bd8950387213e5a8bf5d86daef284ad8b9c31fb69e43d70253eca40fdb3fd0dad5ea27c7288dc9f4910ffcd7cce1; s_v_web_id=verify_l5rpgfoe_2FbyEP85_x7NT_4lnS_8mmC_nBHk2mpOAmBm; passport_csrf_token=512069ecb5db0a00d52dbef10af4dc80; passport_csrf_token_default=512069ecb5db0a00d52dbef10af4dc80; strategyABtestKey=1659508208.59; download_guide=%221%2F20220803%22; THEME_STAY_TIME=%22299791%22; IS_HIDE_THEME_CHANGE=%221%22; __ac_nonce=062ea61cf0015449b429c; __ac_signature=_02B4Z6wo00f01qBh60AAAIDCIGMRAcbkcOKgQe.AAMr-cutAXyUweCACuMTMehP6MqIdwv2UEAokF6bgwnKJHr-hyQGeTR3ihhKs3ZnrQMVRw201.2SFchUf9IYC773W0pl8PgdRzDjOMuyxe4; douyin.com; home_can_add_dy_2_desktop=%221%22; tt_scid=JJ6iYRr.ZdTE5Je0lI2iTvxgISufJnQJCWA52nAk039mUVI2M0tFOAEhXUTbS5dR4460; msToken=YBz-BW7m8hhFhPyQMrwofvkqhXmfcY5CZz5CWtGWgAklgELhEww4OWk067p3bI2IksUw7vbX7uKG2jb1niFtISXpdv-vue9YY6lBHXOIkPQdQg9oJVeQs2s=; msToken=B66A2vCUlfhY3aX_Vf9_z5Lk-SME5-nNGQXkKPOAJM42SYQsWlg9qUtM2Hr6xw4rZYpkGhO7yzt92WXL4nJ3FU2EbimiozujrVjWw-6IWDFLWVfM9np7oSw=',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36'
        }

headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3877.400 QQBrowser/10.8.4506.400',
                'cookie': '__gads=ID=0613c5de4392f6a6-2268f52184cf0004:T=1640239783:RT=1640239783:S=ALNI_MYFmzURQ4PZLUsx8kWq5VTByZe82A; Hm_lvt_338f36c03fc36a54e79fbd2ebdae9589=1640239784,1640259798; Hm_lpvt_338f36c03fc36a54e79fbd2ebdae9589=1640259798'
            }

response = requests.get(url=url, headers=headers)
print(response.text)