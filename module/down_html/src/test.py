from selenium.webdriver.chrome.service import Service
from selenium import webdriver
import time,os


import argparse

# chrome.exe --remote-debugging-port=9222 --user-data-dir=“D:\chromedata” wode  9223 dide
#chrome.exe --remote-debugging-port=9222 --user-data-dir="D:\chromedata" --headless --disable-gpu --no-sandbox --disable-popup-blocking
parser=argparse.ArgumentParser()
parser.add_argument('--ip',type=str,default='127.0.0.1:9223')
parser.add_argument('--isheadless', type=bool, default=False)
parser.add_argument('--isplay', type=bool, default=True)
parser.add_argument('--issave', type=bool, default=False)
args,_=parser.parse_known_args()
def init_driver(args):
    port = args.ip.split(":")[-1].strip()
    if args.isheadless:
        cmd = r'chrome.exe --remote-debugging-port=%s --user-data-dir="D:\chromedata%s" --headless --disable-gpu --no-sandbox --disable-popup-blocking'%(port,port)
    else:
        cmd = r'chrome.exe --remote-debugging-port=%s --user-data-dir="D:\chromedata%s"' % (port, port)
    p = os.popen(cmd)
    time.sleep(1)
    option = webdriver.ChromeOptions()
    # option.add_experimental_option('excludeSwitches', ['enable-automation'])  # 模拟真正浏览器
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
    driver = webdriver.Chrome(executable_path=driver_path,options=option) #
    driver.implicitly_wait(10)
    return driver,driver_service
    # driver.maximize_window()
def open_url(driver,url="https://member.bilibili.com/platform/upload-manager/article?page=1"):
    handles = driver.window_handles
    print(handles)
    driver.get(url)

if __name__ == '__main__':
    try:
        driver,driver_service=init_driver(args)
        open_url(driver)
    except Exception as e:
        print(e)
    finally:
        driver.quit()
        driver_service.stop()
        print("关闭")
