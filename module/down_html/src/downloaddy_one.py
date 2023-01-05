import requests  # 导入requests模块
import re
import os
import json


def dy(txt):

    if not os.path.exists('./downloaded_url'):
        os.makedirs('./downloaded_url')

    if os.path.exists('./downloaded_url/video_url.txt'):
        with open('./downloaded_url/video_url.txt','r') as f:
            lines=f.readlines()
            downloaded_url=set(lines)
    else:
        downloaded_url=set()

    t = re.findall('(https://v.douyin.com/.*?/)', txt, re.S)
    if len(t)!=0:
        html = requests.get(t[0], allow_redirects=False)
        # 获取跳转地址
        h1=html.text
        url2=html.headers['Location']
        # print(url2)
        item_ids = re.findall('video\/(.*?)\/\?region', url2)
        if len(item_ids)!=0:
            item_id=item_ids[0]
            if item_id+'\n' in downloaded_url:
                print('该链接已下载过')
                return
            # video_url,title,headers=download1(item_id=item_ids[0])
            video_url,title,headers=download2(item_id=item_id)

            new_title = re.sub(
                u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a#!~%&:;\(\)?,\"\.，《》：“！？【】——。])", "", title)
            video_response = requests.get(url=video_url, headers=headers)  # 发送下载视频的网络请求
            if video_response.status_code == 200:  # 如果请求成功
                z = os.getcwd()
                temp_path = z + '/有意思的视频/'  # 在程序当前文件夹下建立文件夹
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                data = video_response.content  # 获取返回的视频二进制数据
                c = '%s.mp4' % new_title  # 视频文件的命名
                file = open(temp_path + c, 'wb')  # 创建open对象
                file.write(data)  # 写入数据
                file.close()  # 关闭
                print(title+"视频下载成功！")
            with open('./downloaded_url/video_url.txt', 'a') as f:
                f.write(item_id+'\n')
        else:print('请输入正确的分享链接！')

# while 1:
#     txt = input("请输入抖音分享链接(0退出):")
#     if txt!=str(0):
#         dy(txt)
#     else:
#         print("退出")
#         break

def download1(item_id):
    url = f'https://www.iesdouyin.com/web/api/v2/aweme/iteminfo/?item_ids={item_id}'
    # ur = 'https://www.douyin.com/video/7064550745437146383'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3877.400 QQBrowser/10.8.4506.400',
        'cookie': '__gads=ID=0613c5de4392f6a6-2268f52184cf0004:T=1640239783:RT=1640239783:S=ALNI_MYFmzURQ4PZLUsx8kWq5VTByZe82A; Hm_lvt_338f36c03fc36a54e79fbd2ebdae9589=1640239784,1640259798; Hm_lpvt_338f36c03fc36a54e79fbd2ebdae9589=1640259798'
    }
    html2 = requests.get(url, headers=headers)
    # print(html2)  # 链接成功200
    t2 = html2.json()  # 获取视频所在的整个网页内容
    title = html2.json()['item_list'][0]['desc']
    # print(title)
    video_id = html2.json()['item_list'][0]['video']['play_addr']['uri']
    video_url = f'https://aweme.snssdk.com/aweme/v1/play/?video_id={video_id}&ratio=1080p&line=0'
    # video_url = html2.json()['item_list'][0]['video']['play_addr']['url_list'][0]
    return video_url,title,headers

def download2(item_id):
    url='https://www.douyin.com/video/%s'%item_id
    headers = {
        # 无论是否有登陆 都有 cookie
        'cookie': 'douyin.com; ttwid=1%7CHj5s1edW817SdFa5U5hnifjpPs-xPs4Lv7vC7DQNm60%7C1652797030%7C21e300be59bef00949a7d8f790beeeb4db77670205394cc7cce9dcd0229b5e95; odin_tt=fded742fd6268e282962a1c63dd4f62e37e3bd8950387213e5a8bf5d86daef284ad8b9c31fb69e43d70253eca40fdb3fd0dad5ea27c7288dc9f4910ffcd7cce1; s_v_web_id=verify_l5rpgfoe_2FbyEP85_x7NT_4lnS_8mmC_nBHk2mpOAmBm; passport_csrf_token=512069ecb5db0a00d52dbef10af4dc80; passport_csrf_token_default=512069ecb5db0a00d52dbef10af4dc80; strategyABtestKey=1659508208.59; download_guide=%221%2F20220803%22; THEME_STAY_TIME=%22299791%22; IS_HIDE_THEME_CHANGE=%221%22; __ac_nonce=062ea61cf0015449b429c; __ac_signature=_02B4Z6wo00f01qBh60AAAIDCIGMRAcbkcOKgQe.AAMr-cutAXyUweCACuMTMehP6MqIdwv2UEAokF6bgwnKJHr-hyQGeTR3ihhKs3ZnrQMVRw201.2SFchUf9IYC773W0pl8PgdRzDjOMuyxe4; douyin.com; home_can_add_dy_2_desktop=%221%22; tt_scid=JJ6iYRr.ZdTE5Je0lI2iTvxgISufJnQJCWA52nAk039mUVI2M0tFOAEhXUTbS5dR4460; msToken=YBz-BW7m8hhFhPyQMrwofvkqhXmfcY5CZz5CWtGWgAklgELhEww4OWk067p3bI2IksUw7vbX7uKG2jb1niFtISXpdv-vue9YY6lBHXOIkPQdQg9oJVeQs2s=; msToken=B66A2vCUlfhY3aX_Vf9_z5Lk-SME5-nNGQXkKPOAJM42SYQsWlg9qUtM2Hr6xw4rZYpkGhO7yzt92WXL4nJ3FU2EbimiozujrVjWw-6IWDFLWVfM9np7oSw=',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    # title = re.findall('<title data-react-helmet="true">(.*?)- 抖音</title>', response.text)[0]
    html_data = re.findall('<script id="RENDER_DATA" type="application/json">(.*?)</script', response.text)[0]
    # 解码  ---> requests简单使用 方法 系统课程都教授了
    # requests.utils.unquote 解码  requests.utils.quote 编码
    html_data = requests.utils.unquote(html_data)
    # 转换数据类型
    json_data = json.loads(html_data)
    # print(len(json_data))
    # 字典取值 字典取值, 根据冒号左边的内容, 提取冒号右边的内容
    for k in json_data:
        if 'aweme' in json_data[k]:
            if 'detail' in json_data[k]['aweme']:
                video_url = 'https:' + \
                            json_data[k]['aweme']['detail']['video']['bitRateList'][0]['playAddr'][0]['src']
                title = json_data[k]['aweme']['detail']['desc']
            else:
                raise ValueError('video_url 取值不正确')
    return video_url,title,headers


if __name__ == '__main__':
    address='https://v.douyin.com/rSwao5w/'
    address='https://v.douyin.com/rDhvgv7/'
    address='https://v.douyin.com/rfCkjfR/'
    address='https://v.douyin.com/rfCLbd6/'
    address='https://v.douyin.com/rQ8dYBF/'
    address='https://v.douyin.com/r3YVUAb/'
    address='https://v.douyin.com/h1GDq8B/'
    address='https://v.douyin.com/h1Gy9E3/'
    address='https://v.douyin.com/rEbQXhc/'
    preix="https://v.douyin.com/"
    address_lis=['hmN9N97','hVeQg6a','keraxVk','k8MuMTY','kRykArB','kFeFyoP','kF7YfHt','kLLTsn2','h7uqdu1']
    for d in address_lis:
        address="%s%s/"%(preix,d)
        dy(address)