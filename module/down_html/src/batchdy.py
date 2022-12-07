# -*- coding:utf-8 -*-
# 导入数据请求模块
import requests
# 导入正则
import re
# 导入json
import json
# 导入格式化输出模块
from pprint import pprint
# 导入时间模块
import time

"""
使用 selenium 获取所有视频详情页url地址
    selenium ---> 通过浏览器驱动 ----> 模拟操作浏览器
    - 谷歌浏览器
    - 谷歌驱动 ---> 版本根据浏览器版本来的, 大版本一样 小版本最相近的...
        下载好之后, 可以解压, 
            1. 和你代码一个地方 
            2. python安装目录里面
            3. 写代码的时候指定路径
    selenium 版本 3.141.0

模拟人的行为去操作浏览器
"""
from selenium import webdriver
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default="./有趣的故事")
parser.add_argument('--isheadless', type=bool, default=True)
parser.add_argument('--user_id', type=str,
                    default='MS4wLjABAAAAkzRSrOuSsM4Z1Ricsddumx_aSvX0jmOPcQR2qTs3PEtImBD8BomLrqvtIOBKOL0P')

args, _ = parser.parse_known_args()
args.user_id = "MS4wLjABAAAA8xUmseK9-WQLGOWbjXCpYcJZU0HPGUf9-qOZ1S7oZ0Q"  # 科学旅行号
args.user_id = "MS4wLjABAAAA8Nl-RLXjSF0kleaBbiP5bkEtuck5xzhr5mFCL_ybKTBv6NGM_wDbOS-Q8m5hsLAh"  # 无聊的知识
args.user_id = "MS4wLjABAAAA-wxCgkOlTyeUUENqTmsh6aOLOVOOniShqWtf6lvYNe4fE1GD_K_PvrrCdcBCQH7n"  # 有趣的故事
# args.user_id = "MS4wLjABAAAAM0PAT7Egg1e6KKkmpNXPHoo53ul1BSP_c5GAo-o88D-tkIh__vQAmO5s48iYj4BA"  # 足球

def driver_init(args):
    if args.isheadless:
        cmd = r'chrome.exe --remote-debugging-port=9220 --user-data-dir="D:\chromedataco" --headless --disable-gpu --no-sandbox --disable-popup-blocking'
    else:
        cmd = r'chrome.exe --remote-debugging-port=9220 --user-data-dir="D:\chromedataco"'
    p = os.popen(cmd)
    # print("p.read(): {}\n".format(p.read()))

    # 打开一个浏览器
    option = webdriver.ChromeOptions()
    option.add_experimental_option("debuggerAddress", "127.0.0.1:9220")
    if args.isheadless:
        # 无头模式
        option.add_argument('headless')
        # 沙盒模式运行
        option.add_argument('no-sandbox')
        # 大量渲染时候写入/tmp而非/dev/shm
        option.add_argument('disable-dev-shm-usage')
    driver_path = "../chromedriver"
    driver = webdriver.Chrome(driver_path, options=option)
    return driver

def main(args,driver):
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists('./downloaded_url'):
        os.makedirs('./downloaded_url')

    # 访问网址
    url = "https://www.douyin.com/user/%s" % args.user_id
    driver.get(url)

    if os.path.exists('./downloaded_url/video_url.txt'):
        with open('./downloaded_url/video_url.txt','r') as f:
            lines=f.readlines()
            downloaded_url=set(lines)
    else:
        downloaded_url=set()

    def drop_down():
        for x in range(1, 40, 2):
            time.sleep(1)
            j = x / 9
            js = 'document.documentElement.scrollTop = document.documentElement.scrollHeight * %f' % j
            driver.execute_script(js)


    time.sleep(5)
    print("翻页")
    drop_down()

    # 通过css选择去定位元素 ---> 找标签
    # lis = driver.find_elements_by_css_selector('.ECMy_Zdt')
    # lis = driver.find_elements_by_css_selector('#douyin-right-container > div:nth-child(2) > div > div > div:nth-child(2) > div.mwo84cvf > div.wwg0vUdQ > div.UFuuTZ1P > ul')
    lis = driver.find_elements_by_css_selector('#douyin-right-container > div:nth-child(2) > div > div > div:nth-child(2) > div.mwo84cvf > div.wwg0vUdQ > div.UFuuTZ1P > ul li')
    print("视频链接数:", len(lis))
    # for循环
    video_url_lis=[]
    for idx,li in enumerate(lis):
        try:
            url = li.find_element_by_css_selector('a').get_attribute('href')
            """
            1. 发送请求, 模拟浏览器对于url地址发送请求
                需要注意什么细节:
                    <Response [200]> 表示是请求成功, 但是不代表你得到数据
            """
            # 确定url地址
            # url = 'https://www.douyin.com/video/7064550745437146383'
            item_id = url.split("/")[-1].strip()
            url = f'https://www.iesdouyin.com/web/api/v2/aweme/iteminfo/?item_ids={item_id}'
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
            # video_url = f'https://aweme.snssdk.com/aweme/v1/play/?video_id={video_id}&ratio=1080p&line=0'
            video_url = html2.json()['item_list'][0]['video']['play_addr']['url_list'][0]
            video_url = video_url.replace("playwm","play")
            # html3 = requests.get(video_url, headers=headers)
            # # print(html3.url)
            if video_url+'\n' in downloaded_url:
                print('%s该链接已下载过'%(str(idx)))
                continue
            video_response = requests.get(url=video_url, headers=headers)  # 发送下载视频的网络请求
            if video_response.status_code == 200:  # 如果请求成功
                z = os.getcwd()
                data = video_response.content  # 获取返回的视频二进制数据
                rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
                new_title = re.sub(rstr, "_", title)  # 过滤不能作为文件名的字符，替换为下划线
                title_nolabel = new_title.split("#")[0]
                new_title = '/%s.mp4' % new_title  # 视频文件的命名
                if os.path.exists(save_path + new_title) or os.path.exists(save_path + "_move" + new_title) or os.path.exists(save_path + "_move" + '/%s.mp4' % title_nolabel):
                    print("%s%s 已下载过" % (str(idx),new_title))
                    continue
                file = open(save_path + new_title, 'wb')  # 创建open对象
                file.write(data)  # 写入数据
                file.close()  # 关闭
                print(str(idx) + new_title + " 视频下载成功！")
                with open('./downloaded_url/video_url.txt', 'a') as f:
                    f.write(video_url+'\n')
        except Exception as e:
            print(e)

def get_user_id(url):
    t = re.findall('(https://v.douyin.com/.*?/)', url, re.S)
    if len(t) != 0:
        html = requests.get(t[0], allow_redirects=False)
        # 获取跳转地址
        h1 = html.text
        url = html.headers['Location']
        url =url.split('/')[-1]
    return url
if __name__ == '__main__':
    dic_url={'爆笑':'https://v.douyin.com/hFjyf2B/',
             '有趣的故事':"MS4wLjABAAAA-wxCgkOlTyeUUENqTmsh6aOLOVOOniShqWtf6lvYNe4fE1GD_K_PvrrCdcBCQH7n",
             '有趣的故事 ':"https://v.douyin.com/h8xHqDn/",
             '名人大咖':"https://v.douyin.com/h8xccGx/",
             '名人大咖 ':"https://v.douyin.com/h8QN6Pq/",
             '怀旧故事':'https://v.douyin.com/h8x7pQG/'}
    driver=driver_init(args)
    try:
        for name,url in dic_url.items():
            name=name.strip()
            print("即将下载%s视频..."%name)
            args.user_id=get_user_id(url.strip())
            args.save_path='./'+name
            main(args,driver)
    except:
        os.popen("taskkill /f /t /im chromedriver.exe")
        os.popen("taskkill /f /t /im chrome.exe")
        print("下载失败，然后终止")
        exit(0)
    os.popen("taskkill /f /t /im chromedriver.exe")
    os.popen("taskkill /f /t /im chrome.exe")
    print("下载全部成功，然后终止")
