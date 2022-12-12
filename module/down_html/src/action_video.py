from selenium import webdriver
import pathlib
import time, datetime
import shutil, os, re
from selenium.webdriver.common.keys import Keys

import argparse
from selenium.webdriver.chrome.service import Service

# chrome.exe --remote-debugging-port=9222 --user-data-dir=“D:\chromedata” wode  9223 dide
# chrome.exe --remote-debugging-port=9222 --user-data-dir="D:\chromedata" --headless --disable-gpu --no-sandbox --disable-popup-blocking
parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1:9121')
parser.add_argument('--isheadless', type=int, default=1)
parser.add_argument('--isplay', type=int, default=0)
parser.add_argument('--issave', type=int, default=1)

args, _ = parser.parse_known_args()

print(args)

def init_driver(args):
    port = args.ip.split(":")[-1].strip()
    chrome_dir = r"D:\chromedata%s" % (port)

    if not os.path.exists(chrome_dir):
        os.makedirs(chrome_dir)
    if args.isheadless:
        cmd = r'chrome.exe --remote-debugging-port=%s --user-data-dir="D:\chromedata%s" --headless --disable-gpu --no-sandbox --disable-popup-blocking' % (
        port, port)
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
    # 控制chromedriver服务
    driver_service = Service("../chromedriver")
    driver_service.command_line_args()
    driver_service.start()  # 开启一个chromedriver.exe任务
    driver = webdriver.Chrome(executable_path=driver_path, options=option)  #
    driver.implicitly_wait(10)
    return driver, driver_service
    # driver.maximize_window()


def open_url(driver, url="https://member.bilibili.com/platform/upload-manager/article?page=1"):
    driver.get(url)
    return


def save_video_url(driver, file):
    try:
        driver.find_element_by_xpath('//*[@id="cc-body"]/div[2]/div[2]/div[2]/div[1]/div[2]/div[2]/a').click()
        time.sleep(1)
        lis = driver.find_elements_by_css_selector(
            '#cc-body > div.cc-content-body.upload-manage > div.article-v2-wrap.content > div.is-article.cc-article-wrp > div:nth-child(2) > div.article-list_wrap > div')
        video_url_lis = []
        for li in lis:
            # element=li.find_element_by_xpath('//*[@title="播放"]//span[contains(@class,click-text)]')
            # s=element.get_attribute('outerHTML')
            # play_num=int(re.findall('>(\d+)<', s)[0])
            # if play_num<=10:
            # li.find_element_by_xpath("//*[@class='more-btn']").click()
            video_url = li.find_element_by_css_selector('a').get_attribute('href')
            print(video_url)
            video_url_lis.append(video_url)
    except Exception as e:
        print(e)
        return
    file.write('\n'.join(video_url_lis) + '\n')


def to_sec(duration):
    '''
    :param duration: '00:14'
    :return:
    '''
    tmp = duration.split(':')
    duration_int = 0
    for idx, i in enumerate(tmp[::-1]):
        if idx == 1:
            lv = 60
        elif idx == 0:
            lv = 1
        elif idx == 2:
            lv = 3600
        else:
            ValueError('time format is error')

        duration_int += int(i) * lv
    return duration_int


def play_video(driver, url):
    driver.get(url)

    tt = driver.find_element_by_xpath(
        '//*[contains(@class,"time-duration")]').is_displayed()  # false表示被隐藏，不能通过elemment.text获取文本
    duration = driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("textContent")
    # duration=driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("innerText")
    # duration=driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("innerHTML")

    duration = to_sec(duration)
    print(duration)
    time.sleep(duration * 0.8)
    element = driver.find_element_by_xpath('// * // span[ @ title = "点赞（Q）"]')
    if element.get_attribute("class") == "like":
        element.click()
    time.sleep(duration * 0.2)
    return duration


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


def main(args):
    print(datetime.datetime.now().strftime('%Y年%m月%d号 %H点%M分'))
    ip_lis = ['127.0.0.1:9125',"127.0.0.1:9122", '127.0.0.1:9123', '127.0.0.1:9124'] #zuqiu wode dide huangde
    if args.issave:
        file = open('./videoB_url.txt', 'w')
        for ip in ip_lis:
            try:
                args.ip = ip
                driver, driver_service = init_driver(args)
                open_url(driver)
                save_video_url(driver, file)
                print('ip为：%s的视频链接下载完毕' % ip)
                driver.quit()
                driver_service.stop()
                pid = get_pid(args)
                if pid:
                    os.popen("taskkill /pid %s -t -f" % pid)
                    print("%s的进程被杀死" % args.ip)
            except Exception as e:
                print(e)
                file.close()
                driver.quit()
                driver_service.stop()
                for ip in ip_lis:
                    args.ip = ip
                    pid = get_pid(args)
                    if pid:
                        os.popen("taskkill /pid %s -t -f" % pid)
                    else:
                        print("%s 的进程没有杀死" % args.ip)
                print("下载失败退出")
                exit()
        file.close()
        print("下载成功退出")

    if args.isplay:
        print("start play...")
        try:
            with open('./videoB_url.txt', 'r') as f:
                lines = f.readlines()
            driver, driver_service = init_driver(args)
            duration_sum = 0
            for idx, url in enumerate(lines):
                if url.strip() == '': continue
                duration = play_video(driver, url)
                duration_sum += duration
                print("ip为：%s，第%s篇播放完毕" % (args.ip, idx + 1))
            print("ip为:%s，总共观看时长%s秒" % (args.ip, duration_sum))
        except Exception as e:
            print(e)
            driver.quit()
            driver_service.stop()
            pid = get_pid(args)
            if pid:
                os.popen("taskkill /pid %s -t -f" % pid)
            else:
                print("%s 的进程没有杀死" % args.ip)
            print("自动播放失败退出")
        driver.quit()
        driver_service.stop()
        pid = get_pid(args)
        if pid:
            os.popen("taskkill /pid %s -t -f" % pid)
        else:
            print("%s 的进程没有杀死" % args.ip)
        print("自动播放成功退出")


if __name__ == '__main__':
    main(args)
