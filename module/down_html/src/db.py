import selenium
from selenium import webdriver
import pathlib
import time, datetime
import shutil, os, re
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service

import argparse, random

# chrome.exe --remote-debugging-port=9222 --user-data-dir=“D:\chromedata” wode  9223 dide
# chrome.exe --remote-debugging-port=9222 --user-data-dir="D:\chromedata" --headless --disable-gpu --no-sandbox --disable-popup-blocking
parser = argparse.ArgumentParser()
parser.add_argument('--video_addr', type=str, default="/tjl02")
parser.add_argument('--video_label', type=str, default='label1,label2')
parser.add_argument('--ip', type=str, default='127.0.0.1:9225')
parser.add_argument('--video_describe', type=str, default='视频')
parser.add_argument('--isheadless', type=int, default=0)
parser.add_argument('--num', type=int, default=1)

args, _ = parser.parse_known_args()
cate1 = "知识"
cate2 = "人文历史"
if args.video_addr == 'gaoxiao':
    args.video_addr = '/爆笑'
    cate1 = "生活"
    cate2 = "搞笑"
if args.video_addr == 'minren':
    args.video_addr = '/名人大咖'
    cate1 = "知识"
    cate2 = "社科·法律·心理"
if args.video_addr == 'huaijiu':
    args.video_addr = '/怀旧故事'
if args.video_addr == 'sense': args.video_addr = '/有意思的视频'
if args.video_addr == 'santi': args.video_addr = '/三体'
if args.video_addr == 'youqu': args.video_addr = '/有趣的故事'
if args.video_addr == 'suiji': args.video_addr = random.choice(["/有意思的视频",'/名人大咖'])
if args.video_addr == 'gaoxiaozuqiu': args.video_addr = '/搞笑足球'
if args.video_addr == 'zuqiu': args.video_addr = '/足球'
if args.video_addr == 'lol': args.video_addr = '/LOL'
if args.video_addr == 'movie': args.video_addr = '/电影解说'

pwd_dir = os.getcwd()
# print("pwd_dir:", pwd_dir)

args.video_label = args.video_addr.split('/')[-1].strip()
if args.video_addr.startswith('/tjl') or args.video_addr.startswith('./tjl'): args.video_label = '名人大咖#胖东来#刘强东#王健林#马云'

args.video_addr = pwd_dir + args.video_addr


move_dir = args.video_addr + "_move"

if not os.path.exists(move_dir):
    os.makedirs(move_dir)


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
    time.sleep(5)
    # driver.maximize_window()

    driver.implicitly_wait(10)
    return driver,driver_service

def publish_bilibili(args, driver, path_mp4):
    '''
     作用：发布b站视频
    '''

    # 进入创作者页面，并上传视频
    # driver.refresh()
    driver.get("https://member.bilibili.com/platform/upload/video/frame")
    title = path_mp4.split('\\')[-1].split("#")[0]
    title=title.strip().replace('.mp4','')
    label = path_mp4.split('.')[0].split('\\')[-1].split("#")[1:]
    label = [re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", l) for l in label if
             "dou" not in l and "抖音" not in l and len(l.strip()) > 1]

    label = '#'.join(label)
    if label: args.video_label = label
    args.video_describe = args.video_label
    if len(title) <= 6 and len(args.video_label)>len(title): title = args.video_label

    try:
        alert = driver.switch_to.alert()
        print('alert text:', alert.text)
        alert.accept()
    except:
        pass
    time.sleep(3)
    try:
        driver.switch_to.frame(driver.find_element_by_xpath('//iframe[@name="videoUpload"]'))
    except Exception as e:
        pass
    # print(path_mp4)
    driver.find_element_by_xpath('//input[@type="file" and contains(@accept,"mp4")]').send_keys(path_mp4)

    # 等待视频上传完成
    idx = 0
    while True:
        try:
            idx += 1
            driver.find_element_by_xpath('//*[text()="上传完成"]')
            break;
        except Exception as e:
            print("视频还在上传中···")
            if idx > 100:
                ValueError("视频上传超时")
                break

    print("视频已上传完成！")

    # # 添加封面
    # driver.find_element_by_xpath('//*[text()="更改封面"]').click()
    # element=driver.find_element_by_xpath('//*[@class="cover-cut"]//*[contains(text(),"完成")]')
    # driver.execute_script("arguments[0].click();", element)
    # # time.sleep(1)
    # driver.find_element_by_xpath('//div[text()="图片上传"]').click()
    # # time.sleep(1)
    # driver.find_element_by_xpath('//input[@type="file"]').send_keys(path_cover)
    # time.sleep(3)
    # driver.find_element_by_xpath('//*[text()="确定"]').click()

    # 输入标题

    try:
        driver.find_element_by_xpath('//input[contains(@placeholder,"标题")]').clear()
        time.sleep(1)
        driver.find_element_by_xpath('//input[contains(@placeholder,"标题")]').send_keys(title)  # 如果能执行就重新输入，否则不输入
    except:
        pass
    # 选择分类
    try:
        element = driver.find_element_by_xpath('//*[contains(@class,"select-container")]')
        # element = driver.find_element_by_xpath('//*[@id="video-up-app"]/div[2]/div/div/div[1]/div[2]/div[5]/div/div[2]/div/div')
        # print(element.get_attribute("class"))
        driver.execute_script("arguments[0].click();", element)

        time.sleep(1)
        driver.find_element_by_xpath('//*[@class="f-item-content" and text()="{}"]'.format(cate1)).click()
        time.sleep(1)
        driver.find_element_by_xpath('//*[@class="item-main" and text()="{}"]'.format(cate2)).click()
    except Exception as e:
        print('分区选择失败，使用默认的: ', e)

    # 选择标签
    time.sleep(2)
    try:
        element=driver.find_element_by_xpath(
            '//*[text()="参与话题："]/..//*[@class="tag-topic-list"]/span[1]//*[@class="hot-tag-item"]')
        driver.execute_script("arguments[0].click();", element)
    except Exception as e:
        print(e)
    for label in args.video_label.strip().split('#'):
        print("lable:", label)
        driver.find_element_by_xpath('//input[@placeholder="按回车键Enter创建标签"]').send_keys(label)
        driver.find_element_by_xpath('//input[@placeholder="按回车键Enter创建标签"]').send_keys(Keys.ENTER)
        time.sleep(1)

    time.sleep(1)

    # 输入描述
    driver.find_element_by_xpath('//*[@editor_id="desc_at_editor"]//br').send_keys(args.video_describe)
    # 刚开始可以先注释掉发布，人工进行检查内容是否有问题
    # time.sleep(3)
    # 点击发布
    # driver.find_element_by_xpath('//button[text()="立即投稿"]').click()
    # driver.find_element_by_xpath('//*[@id="video-up-app"]/div[2]/div/div/div[1]/div[3]/div[15]/div/span').click()
    time.sleep(3)
    try:
        driver.find_element_by_xpath('//*[@class="submit-add" and text()="立即投稿"]').click()
    except:
        element = driver.find_element_by_xpath('//*[@class="submit-add" and text()="立即投稿"]')
        driver.execute_script("arguments[0].click();", element)
    print("投稿成功")
    time.sleep(3)


def main(args,driver):
    # 基本信息
    # 视频存放路径
    catalog_mp4 = args.video_addr
    # 视频描述
    # time.sleep(10)


    path = pathlib.Path(catalog_mp4)
    file_path=list(path.iterdir())
    # random.shuffle(file_path)

    if not os.path.exists('./uploaded/'):
        os.makedirs('./uploaded/')
    try:
        with open('./uploaded/'+ args.ip.split(':')[-1].strip()+'.txt','r') as f:
            lines=f.readlines()
            title_set=set([u.strip() for u in lines ])
    except:
        title_set={}
    # 视频地址获取
    path_mp4 = ""
    idx = 0
    for i in file_path:
        if (".mp4" in str(i)):
            path_mp4 = str(i)
        else:
            continue
        title = path_mp4.split('\\')[-1].split("#")[0]
        if title in title_set:
            os.remove(path_mp4)
            continue
        print("检查到视频路径：" + path_mp4)
        publish_bilibili(args, driver, path_mp4)
        try:
            shutil.move(path_mp4, move_dir)
        except Exception as e:
            print(e)
        with open('./uploaded/' + args.ip.split(':')[-1].strip() + '.txt', 'a') as f:
            f.write(title+'\n')
        idx += 1
        if idx > args.num - 1: break
    if idx==0:ValueError("视频目录为空，请先下载视频")
    # 封面地址获取
    # path_cover = ""
    # for i in path.iterdir():
    #     if (".png" in str(i) or ".jpg" in str(i)):
    #         path_cover = str(i);
    #         break;
    #
    # if (path_cover != ""):
    #     print("检查到封面路径：" + path_cover)
    # else:
    #     print("未检查到封面路径，程序终止！")
    #     exit()


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
    # 开始执行b站视频发布


if __name__ == '__main__':
    # shutil.move(r'D:\code\pycharm\project1\github\NLP\module\down_html\src\tjl02\08年最惨痛的春运教训，4万解放军紧急出动（下）.mp4', './tjl02_move/')
    # exit(0)#todo
    print(datetime.datetime.now().strftime('%Y年%m月%d号 %H点%M分'))
    print("ip:", args.ip)
    # if args.ip[-4:]=='9222':exit(0)
    try:
        driver,driver_service=init_driver(args)
        main(args,driver)
    except Exception as e:
        print(e)
        pid = get_pid(args)
        if pid:
            os.popen("taskkill /pid %s -t -f" % pid)
        else:
            print("%s 的进程没有杀死" % args.ip)
        print("投稿失败，然后终止")
        driver.quit()
        driver_service.stop()
        exit(0)
    driver.quit()
    driver_service.stop()
    pid = get_pid(args)
    if pid:
        os.popen("taskkill /pid %s -t -f" % pid)
    else:
        print("%s 的进程没有杀死" % args.ip)
    print("投稿成功，然后终止")
