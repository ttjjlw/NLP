
from selenium import webdriver
import pathlib
import time,datetime
import shutil,os,re
from selenium.webdriver.common.keys import Keys

import argparse

# chrome.exe --remote-debugging-port=9222 --user-data-dir=“D:\chromedata” wode  9223 dide
#chrome.exe --remote-debugging-port=9222 --user-data-dir="D:\chromedata" --headless --disable-gpu --no-sandbox --disable-popup-blocking
parser=argparse.ArgumentParser()
parser.add_argument('--ip',type=str,default='127.0.0.1:9224')
parser.add_argument('--isheadless', type=bool, default=False)
parser.add_argument('--isplay', type=bool, default=True)
parser.add_argument('--issave', type=bool, default=False)

args,_=parser.parse_known_args()

def init_driver():
    port = args.ip.split(":")[-1].strip()
    if args.isheadless:
        cmd = r'chrome.exe --remote-debugging-port=%s --user-data-dir="D:\chromedata%s" --headless --disable-gpu --no-sandbox --disable-popup-blocking'%(port,port)
    else:
        cmd = r'chrome.exe --remote-debugging-port=%s --user-data-dir="D:\chromedata%s"' % (port, port)
    p = os.popen(cmd)
    time.sleep(1)
    option = webdriver.ChromeOptions()
    option.add_experimental_option("debuggerAddress", args.ip)
    driver_path = '../chromedriver'
    if args.isheadless:
        # 无头模式
        option.add_argument('headless')
        # 沙盒模式运行
        option.add_argument('no-sandbox')
        # 大量渲染时候写入/tmp而非/dev/shm
        option.add_argument('disable-dev-shm-usage')
    driver = webdriver.Chrome(executable_path=driver_path,options=option) #
    driver.implicitly_wait(10)
    return driver
    # driver.maximize_window()



def open_url(driver,url="https://member.bilibili.com/platform/upload-manager/article?page=1"):
    driver.get(url)
    driver.get(url)
def save_video_url(driver,downloaded_url):
    driver.find_element_by_xpath('//*[@id="cc-body"]/div[2]/div[2]/div[2]/div[1]/div[2]/div[2]/a').click()
    time.sleep(1)
    lis = driver.find_elements_by_css_selector('#cc-body > div.cc-content-body.upload-manage > div.article-v2-wrap.content > div.is-article.cc-article-wrp > div:nth-child(2) > div.article-list_wrap > div')
    for li in lis:
        # element=li.find_element_by_xpath('//*[@title="播放"]//span[contains(@class,click-text)]')
        # s=element.get_attribute('outerHTML')
        # play_num=int(re.findall('>(\d+)<', s)[0])
        # if play_num<=10:
            # li.find_element_by_xpath("//*[@class='more-btn']").click()
        video_url = li.find_element_by_css_selector('a').get_attribute('href')
        print(video_url)
        if video_url + '\n' in downloaded_url:continue
        with open('./videoB_url.txt', 'a') as f:
            f.write(video_url + '\n')
        # element=li.find_element_by_xpath('//*[@class="meta-title"]')
        # s = element.get_attribute('outerHTML')
        # print(s)
        # video_url=re.findall('href="//(.+/)"', s)[0]
        # print(video_url)
def to_sec(duration):
    '''
    :param duration: '00:14'
    :return:
    '''
    tmp=duration.split(':')
    duration_int=0
    for idx,i in enumerate(tmp[::-1]):
        if idx==1:
            lv=60
        elif idx==0:
            lv=1
        elif idx==2:
            lv=3600
        else:
            ValueError('time format is error')

        duration_int+=int(i)*lv
    return duration_int

def play_video(driver,url):
    driver.get(url)
    element=driver.find_element_by_xpath('// * // span[ @ title = "点赞（Q）"]')
    if element.get_attribute("class")=="like":
        element.click()
    tt=driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').is_displayed()  #false表示被隐藏，不能通过elemment.text获取文本
    print(tt)
    duration=driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("textContent")
    # duration=driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("innerText")
    # duration=driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("innerHTML")

    duration=to_sec(duration)
    print(duration)
    # time.sleep(duration)
    print('s')




if __name__ == '__main__':
    with open('./videoB_url.txt', 'r') as f:
        lines = f.readlines()
        downloaded_url = set(lines)
    try:
        driver=init_driver()
        if args.issave:
            open_url(driver)
            save_video_url(driver,downloaded_url)
        if args.isplay:
            for url in lines:
                play_video(driver,url)
    except Exception as e:
        print(e)
    # os.popen("taskkill /f /t /im chromedriver.exe")
    # os.popen("taskkill /f /t /im chrome.exe")

