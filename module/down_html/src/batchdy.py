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
import datetime
from selenium.webdriver.chrome.service import Service
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
parser.add_argument('--isheadless', type=int, default=0)
parser.add_argument('--istest', type=int, default=0)
parser.add_argument('--ip', type=str, default="127.0.0.1:9220")
parser.add_argument('--user_id', type=str,
                    default='MS4wLjABAAAAkzRSrOuSsM4Z1Ricsddumx_aSvX0jmOPcQR2qTs3PEtImBD8BomLrqvtIOBKOL0P')

args, _ = parser.parse_known_args()
print(args)
args.user_id = "MS4wLjABAAAA8xUmseK9-WQLGOWbjXCpYcJZU0HPGUf9-qOZ1S7oZ0Q"  # 科学旅行号
args.user_id = "MS4wLjABAAAA8Nl-RLXjSF0kleaBbiP5bkEtuck5xzhr5mFCL_ybKTBv6NGM_wDbOS-Q8m5hsLAh"  # 无聊的知识
args.user_id = "MS4wLjABAAAA-wxCgkOlTyeUUENqTmsh6aOLOVOOniShqWtf6lvYNe4fE1GD_K_PvrrCdcBCQH7n"  # 有趣的故事
# args.user_id = "MS4wLjABAAAAM0PAT7Egg1e6KKkmpNXPHoo53ul1BSP_c5GAo-o88D-tkIh__vQAmO5s48iYj4BA"  # 足球

def driver_init(args):
    port = args.ip.split(":")[-1].strip()
    chrome_dir = r"D:\chromedata%s" % (port)

    if not os.path.exists(chrome_dir):
        os.makedirs(chrome_dir)
    if args.isheadless:
        cmd = r'chrome.exe --remote-debugging-port=%s --user-data-dir="D:\chromedata%s" --headless --disable-gpu --no-sandbox --disable-popup-blocking'%(port,port)
    else:
        cmd = r'chrome.exe --remote-debugging-port=%s --user-data-dir="D:\chromedata%s"'%(port,port)
    p = os.popen(cmd)
    # print("p.read(): {}\n".format(p.read()))

    # 打开一个浏览器
    option = webdriver.ChromeOptions()
    option.add_experimental_option("debuggerAddress", args.ip)
    if args.isheadless:
        # 无头模式
        option.add_argument('headless')
        # 沙盒模式运行
        option.add_argument('no-sandbox')
        # 大量渲染时候写入/tmp而非/dev/shm
        option.add_argument('disable-dev-shm-usage')
        # 控制chromedriver服务
    driver_path = "../chromedriver"
    driver_service = Service(driver_path)
    driver_service.command_line_args()
    driver_service.start()  # 开启一个chromedriver.exe任务
    driver = webdriver.Chrome(driver_path, options=option)
    return driver,driver_service

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
            item_id=url.strip().split('/')[-1]
            if item_id+'\n' in downloaded_url:
                print('%s该item_id已下载过'%(str(idx)))
                continue
            """
            1. 发送请求, 模拟浏览器对于url地址发送请求
                需要注意什么细节:
                    <Response [200]> 表示是请求成功, 但是不代表你得到数据
            """
            # 确定url地址
            # url = 'https://www.douyin.com/video/7064550745437146383'
            # item_id = url.split("/")[-1].strip()
            # url = f'https://www.iesdouyin.com/web/api/v2/aweme/iteminfo/?item_ids={item_id}'
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3877.400 QQBrowser/10.8.4506.400',
                'cookie': '__gads=ID=0613c5de4392f6a6-2268f52184cf0004:T=1640239783:RT=1640239783:S=ALNI_MYFmzURQ4PZLUsx8kWq5VTByZe82A; Hm_lvt_338f36c03fc36a54e79fbd2ebdae9589=1640239784,1640259798; Hm_lpvt_338f36c03fc36a54e79fbd2ebdae9589=1640259798'
            }
            headers = {
                # 无论是否有登陆 都有 cookie
                'cookie': 'douyin.com; ttwid=1%7CHj5s1edW817SdFa5U5hnifjpPs-xPs4Lv7vC7DQNm60%7C1652797030%7C21e300be59bef00949a7d8f790beeeb4db77670205394cc7cce9dcd0229b5e95; odin_tt=fded742fd6268e282962a1c63dd4f62e37e3bd8950387213e5a8bf5d86daef284ad8b9c31fb69e43d70253eca40fdb3fd0dad5ea27c7288dc9f4910ffcd7cce1; s_v_web_id=verify_l5rpgfoe_2FbyEP85_x7NT_4lnS_8mmC_nBHk2mpOAmBm; passport_csrf_token=512069ecb5db0a00d52dbef10af4dc80; passport_csrf_token_default=512069ecb5db0a00d52dbef10af4dc80; strategyABtestKey=1659508208.59; download_guide=%221%2F20220803%22; THEME_STAY_TIME=%22299791%22; IS_HIDE_THEME_CHANGE=%221%22; __ac_nonce=062ea61cf0015449b429c; __ac_signature=_02B4Z6wo00f01qBh60AAAIDCIGMRAcbkcOKgQe.AAMr-cutAXyUweCACuMTMehP6MqIdwv2UEAokF6bgwnKJHr-hyQGeTR3ihhKs3ZnrQMVRw201.2SFchUf9IYC773W0pl8PgdRzDjOMuyxe4; douyin.com; home_can_add_dy_2_desktop=%221%22; tt_scid=JJ6iYRr.ZdTE5Je0lI2iTvxgISufJnQJCWA52nAk039mUVI2M0tFOAEhXUTbS5dR4460; msToken=YBz-BW7m8hhFhPyQMrwofvkqhXmfcY5CZz5CWtGWgAklgELhEww4OWk067p3bI2IksUw7vbX7uKG2jb1niFtISXpdv-vue9YY6lBHXOIkPQdQg9oJVeQs2s=; msToken=B66A2vCUlfhY3aX_Vf9_z5Lk-SME5-nNGQXkKPOAJM42SYQsWlg9qUtM2Hr6xw4rZYpkGhO7yzt92WXL4nJ3FU2EbimiozujrVjWw-6IWDFLWVfM9np7oSw=',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
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
                        #video_url的形式，目前用的是第三种
                        # 1、'https://aweme.snssdk.com/aweme/v1/play/?video_id=https://sf3-cdn-tos.douyinstatic.com/obj/ies-music/1651073949424696.mp3&ratio=720p&line=0'
                        # 2、'https://aweme.snssdk.com/aweme/v1/play/?video_id=v0300fg10000cegoaa3c77uem4am2dm0&ratio=720p&line=0'
                        # 3、'https://v26-web.douyinvod.com/c7e88fe2620a5b716e9b0c57e387ecbc/63a9cea3/video/tos/cn/tos-cn-ve-15c001-alinc2/4b0c31c5c0c24392ac46bfbc9b00555a/?a=6383&ch=26&cr=3&dr=0&lr=all&cd=0%7C0%7C0%7C3&cv=1&br=2642&bt=2642&cs=0&ds=4&ft=LjhJEL998xl8uEemg0P5H4eaciDXtQH-95QEeR~q6KLD1Ini&mime_type=video_mp4&qs=0&rc=NWg4Z2c6NGU2aGY3M2VmPEBpam8zdjg6ZjQzZjMzNGkzM0AtYC4yXzY0XmExXzRiXzM2YSNkcWhzcjRnXm5gLS1kLS9zcw%3D%3D&l=2022122623303323F7F280CBF7BB3025A0&btag=38000'
                        title=json_data[k]['aweme']['detail']['desc']
                    else:
                        raise ValueError('video_url 取值不正确')
            # print(title)
            # print(video_url)
            new_title = re.sub(
                u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a#!~%&:;\(\)?,\"\.，《》：“！？【】——。])", "", title)
            if new_title.strip() == '': continue
            # if title_is_exists(new_title,idx,save_path):continue
            video_response = requests.get(url=video_url, headers=headers)  # 发送下载视频的网络请求
            if video_response.status_code == 200:  # 如果请求成功
                z = os.getcwd()
                data = video_response.content  # 获取返回的视频二进制数据
                rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'

                file = open(save_path + new_title, 'wb')  # 创建open对象
                file.write(data)  # 写入数据
                file.close()  # 关闭
                print(str(idx) + new_title + " 视频下载成功！")
                with open('./downloaded_url/video_url.txt', 'a') as f:
                    f.write(item_id+'\n')
        except Exception as e:
            print(e)
            # exit(1) 不能退出

def title_is_exists(new_title,idx,save_path):

    title_nolabel = new_title.split("#")[0]
    new_title = '/%s.mp4' % new_title  # 视频文件的命名
    if os.path.exists(save_path + new_title) or os.path.exists(save_path + "_move" + new_title) or os.path.exists(
            save_path + "_move" + '/%s.mp4' % title_nolabel):
        print("%s%s 已下载过" % (str(idx), new_title))
        return True
    else:
        return False
def get_user_id(url):
    t = re.findall('(https://v.douyin.com/.*?/)', url, re.S)
    if len(t) != 0:
        html = requests.get(t[0], allow_redirects=False)
        # 获取跳转地址
        h1 = html.text
        url = html.headers['Location']
        url =url.split('/')[-1]
        # url = 'MS4wLjABAAAA8xUmseK9-WQLGOWbjXCpYcJZU0HPGUf9-qOZ1S7oZ0Q'
    return url
def get_pid(args):
    p = os.popen("netstat -ano|findstr %s" % args.ip.split(":")[-1].strip())
    data = p.read()
    lines = data.split('\n')
    line_t = [line.strip().split('    ') for line in lines]
    result = []
    for l in line_t:
        result.append([i.strip() for i in l if len(i.strip()) > 0])
    for s in result:
        if s[1] == args.ip:
            pid = s[-1]
            print('pid:', pid)
            return pid
    return None
if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y年%m月%d号 %H点%M分'))
    dic_url={
        '三体': "https://v.douyin.com/h9uWDc5/",
        '搞笑足球': "https://v.douyin.com/hAGB8EA/",
        '名人大咖  ': "https://v.douyin.com/hrvcbE6/",
        '爆笑':'https://v.douyin.com/hFjyf2B/',
             '有趣的故事':"MS4wLjABAAAA-wxCgkOlTyeUUENqTmsh6aOLOVOOniShqWtf6lvYNe4fE1GD_K_PvrrCdcBCQH7n",
             '有趣的故事 ':"https://v.douyin.com/h8xHqDn/",
             '有意思的视频 ':"https://v.douyin.com/h51pSb4/",
              '有意思的视频  ':'https://v.douyin.com/hrvcbE6/https://v.douyin.com/hrvcbE6/',
             '名人大咖':"https://v.douyin.com/h8xccGx/",
             '名人大咖 ':"https://v.douyin.com/h8QN6Pq/",
             '足球':"https://v.douyin.com/hYuvxEm/",
             'LOL':"https://v.douyin.com/h2yWMCP/",
             '怀旧故事':'https://v.douyin.com/h8x7pQG/'}
    if args.istest:
        dic_url={'搞笑足球  ': "https://v.douyin.com/hAGB8EA/"}
    #  '电影解说':"https://v.douyin.com/hNxCfns/"
    driver,driver_service=driver_init(args)
    try:
        for name,url in dic_url.items():
            name=name.strip()
            print("即将下载%s视频..."%name)
            args.user_id=get_user_id(url.strip())
            args.save_path='./'+name
            main(args,driver)
    except Exception as e:
        print(e)
        driver.quit()
        driver_service.stop()
        pid = get_pid(args)
        if pid:
            os.popen("taskkill /pid %s -t -f" % pid)
        else:
            print("%s 的进程没有杀死" % args.ip)
        print("下载失败，然后终止")
        exit(0)
    driver.quit()
    driver_service.stop()
    pid = get_pid(args)
    if pid:
        os.popen("taskkill /pid %s -t -f" % pid)
    else:
        print("%s 的进程没有杀死" % args.ip)
    print("下载全部成功，然后终止")
