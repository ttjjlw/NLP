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

parser=argparse.ArgumentParser()
parser.add_argument('--save_path',type=str,default="../space")
parser.add_argument('--user_id',type=str,default='MS4wLjABAAAAkzRSrOuSsM4Z1Ricsddumx_aSvX0jmOPcQR2qTs3PEtImBD8BomLrqvtIOBKOL0P')


args,_=parser.parse_known_args()
args.user_id="MS4wLjABAAAA8xUmseK9-WQLGOWbjXCpYcJZU0HPGUf9-qOZ1S7oZ0Q"
save_path= args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 打开一个浏览器
driver = webdriver.Chrome('.././chromedriver')
# 访问网址
url="https://www.douyin.com/user/%s"%args.user_id
driver.get(url)

print("翻页")
def drop_down():
    for x in range(1, 40, 2):
        time.sleep(1)
        j = x / 9
        js = 'document.documentElement.scrollTop = document.documentElement.scrollHeight * %f' % j
        driver.execute_script(js)


drop_down()

# 通过css选择去定位元素 ---> 找标签
# lis = driver.find_elements_by_css_selector('.ECMy_Zdt')
# lis = driver.find_elements_by_css_selector('#douyin-right-container > div:nth-child(2) > div > div > div:nth-child(2) > div.mwo84cvf > div.wwg0vUdQ > div.UFuuTZ1P > ul')
lis = driver.find_elements_by_css_selector('#douyin-right-container > div:nth-child(2) > div > div > div:nth-child(2) > div.mwo84cvf > div.wwg0vUdQ > div.UFuuTZ1P > ul li')
# for循环
for li in lis:
    try:
        url = li.find_element_by_css_selector('a').get_attribute('href')
        """
        1. 发送请求, 模拟浏览器对于url地址发送请求
            需要注意什么细节:
                <Response [200]> 表示是请求成功, 但是不代表你得到数据
        """
        # 确定url地址
        # url = 'https://www.douyin.com/video/7064550745437146383'
        # 模拟浏览器 伪装python代码 设置请求头参数
        headers = {
            # 无论是否有登陆 都有 cookie
            'cookie': 'douyin.com; ttwid=1%7CHj5s1edW817SdFa5U5hnifjpPs-xPs4Lv7vC7DQNm60%7C1652797030%7C21e300be59bef00949a7d8f790beeeb4db77670205394cc7cce9dcd0229b5e95; odin_tt=fded742fd6268e282962a1c63dd4f62e37e3bd8950387213e5a8bf5d86daef284ad8b9c31fb69e43d70253eca40fdb3fd0dad5ea27c7288dc9f4910ffcd7cce1; s_v_web_id=verify_l5rpgfoe_2FbyEP85_x7NT_4lnS_8mmC_nBHk2mpOAmBm; passport_csrf_token=512069ecb5db0a00d52dbef10af4dc80; passport_csrf_token_default=512069ecb5db0a00d52dbef10af4dc80; strategyABtestKey=1659508208.59; download_guide=%221%2F20220803%22; THEME_STAY_TIME=%22299791%22; IS_HIDE_THEME_CHANGE=%221%22; __ac_nonce=062ea61cf0015449b429c; __ac_signature=_02B4Z6wo00f01qBh60AAAIDCIGMRAcbkcOKgQe.AAMr-cutAXyUweCACuMTMehP6MqIdwv2UEAokF6bgwnKJHr-hyQGeTR3ihhKs3ZnrQMVRw201.2SFchUf9IYC773W0pl8PgdRzDjOMuyxe4; douyin.com; home_can_add_dy_2_desktop=%221%22; tt_scid=JJ6iYRr.ZdTE5Je0lI2iTvxgISufJnQJCWA52nAk039mUVI2M0tFOAEhXUTbS5dR4460; msToken=YBz-BW7m8hhFhPyQMrwofvkqhXmfcY5CZz5CWtGWgAklgELhEww4OWk067p3bI2IksUw7vbX7uKG2jb1niFtISXpdv-vue9YY6lBHXOIkPQdQg9oJVeQs2s=; msToken=B66A2vCUlfhY3aX_Vf9_z5Lk-SME5-nNGQXkKPOAJM42SYQsWlg9qUtM2Hr6xw4rZYpkGhO7yzt92WXL4nJ3FU2EbimiozujrVjWw-6IWDFLWVfM9np7oSw=',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36'
        }
        # 发送请求  左边url是get请求方法里面形式参数 右边url是我们自定义变量, 传入进去参数
        response = requests.get(url=url, headers=headers)
        # 打印响应对象 <Response [200]>
        t=response.text
        # 2. 获取数据 print(response.text)
        """
        3. 解析数据 ---> 根据返回数据类型<内容> 以及 你想要数据内容 选择最方便合适解析方法
            re正则表示式 直接可以对于字符串数据进行提取

        除非说我们用re提取说出来之后, 是json数据格式, 然后通过数据类型转换之后, 再用字典取值
        """
        # 提取标题
        title = re.findall('<title data-react-helmet="true">(.*?)- 抖音</title>', response.text)[0]
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        title = re.sub(rstr, "_", title)  # 过滤不能作为文件名的字符，替换为下划线
        title =title.strip()
        #
        html_data = re.findall('<script id="RENDER_DATA" type="application/json">(.*?)</script', response.text)[0]
        # 解码  ---> requests简单使用 方法 系统课程都教授了
        # requests.utils.unquote 解码  requests.utils.quote 编码
        html_data = requests.utils.unquote(html_data)
        # 转换数据类型
        json_data = json.loads(html_data)
        print(len(json_data))
        # 字典取值 字典取值, 根据冒号左边的内容, 提取冒号右边的内容
        for k in json_data:
            if'aweme' in json_data[k]:
                if 'detail' in json_data[k]['aweme']:
                    video_url = 'https:' + json_data[k]['aweme']['detail']['video']['bitRateList'][0]['playAddr'][0]['src']
                else:
                    raise ValueError('video_url 取值不正确')
        print(title)
        print(video_url)
        # 保存数据
        video_content = requests.get(url=video_url, headers=headers).content
        with open(save_path+'/' + title + '.mp4', mode='wb') as f:
            f.write(video_content)
    except Exception as e:
        print(e)