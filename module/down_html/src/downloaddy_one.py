import requests  # 导入requests模块
import re
import os


def dy(txt):

    t = re.findall('(https://v.douyin.com/.*?/)', txt, re.S)
    if len(t)!=0:
        html = requests.get(t[0], allow_redirects=False)
        # 获取跳转地址
        h1=html.text
        url2=html.headers['Location']
        # print(url2)
        item_ids = re.findall('video\/(.*?)\/\?region', url2)
        if len(item_ids)!=0:
            ur=f'https://www.iesdouyin.com/web/api/v2/aweme/iteminfo/?item_ids={item_ids[0]}'
            # ur = 'https://www.douyin.com/video/7064550745437146383'
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3877.400 QQBrowser/10.8.4506.400',
                'cookie': '__gads=ID=0613c5de4392f6a6-2268f52184cf0004:T=1640239783:RT=1640239783:S=ALNI_MYFmzURQ4PZLUsx8kWq5VTByZe82A; Hm_lvt_338f36c03fc36a54e79fbd2ebdae9589=1640239784,1640259798; Hm_lpvt_338f36c03fc36a54e79fbd2ebdae9589=1640259798'
            }
            html2 = requests.get(ur,headers=headers)
            # print(html2)  # 链接成功200
            t2=html2.json()  #获取视频所在的整个网页内容
            title=html2.json()['item_list'][0]['desc']
            # print(title)
            video_id=html2.json()['item_list'][0]['video']['play_addr']['uri']
            video_url=f'https://aweme.snssdk.com/aweme/v1/play/?video_id={video_id}&ratio=1080p&line=0'
            # video_url = html2.json()['item_list'][0]['video']['play_addr']['url_list'][0]
            html3=requests.get(video_url,headers=headers)
            #print(html3.url)

            video_response = requests.get(url=video_url, headers=headers)  # 发送下载视频的网络请求
            if video_response.status_code == 200:  # 如果请求成功
                z = os.getcwd()
                temp_path = z + '/抖音视频/'  # 在程序当前文件夹下建立文件夹
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                data = video_response.content  # 获取返回的视频二进制数据
                rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
                new_title = re.sub(rstr, "_", title)  # 过滤不能作为文件名的字符，替换为下划线
                c = '%s.mp4' % new_title  # 视频文件的命名
                file = open(temp_path + c, 'wb')  # 创建open对象
                file.write(data)  # 写入数据
                file.close()  # 关闭
                print(title+"视频下载成功！")
        else:print('请输入正确的分享链接！')

# while 1:
#     txt = input("请输入抖音分享链接(0退出):")
#     if txt!=str(0):
#         dy(txt)
#     else:
#         print("退出")
#         break

if __name__ == '__main__':
    address='https://v.douyin.com/rSwao5w/'
    address='https://v.douyin.com/rDhvgv7/'
    address='https://v.douyin.com/rfCkjfR/'
    address='https://v.douyin.com/rfCLbd6/'
    address='https://v.douyin.com/rQ8dYBF/'
    address='https://v.douyin.com/r3YVUAb/'
    dy(address)