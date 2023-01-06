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
parser.add_argument('--ip', type=str, default='127.0.0.1:9123')
parser.add_argument('--isheadless', type=int, default=1)
parser.add_argument('--istest', type=int, default=1)
parser.add_argument('--isplay', type=int, default=0)
parser.add_argument('--issave', type=int, default=0)
parser.add_argument('--isgetdata', type=int, default=1)

args, _ = parser.parse_known_args()
if args.istest:args.isheadless=0
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
    driver.get("https://member.bilibili.com/platform/upload-manager/article?page=1")
    time.sleep(1)
    driver.get(url)

    tt = driver.find_element_by_xpath(
        '//*[contains(@class,"time-duration")]').is_displayed()  # false表示被隐藏，不能通过elemment.text获取文本
    duration = driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("textContent")
    # duration=driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("innerText")
    # duration=driver.find_element_by_xpath('//*[contains(@class,"time-duration")]').get_attribute("innerHTML")

    duration = to_sec(duration)
    print(duration)
    play_rate=1
    if duration>30:
        play_rate=0.5
        elem=driver.find_element_by_xpath('//*[@class="bpx-player-ctrl-playbackrate-menu-item " and @data-value="2"]')
        driver.execute_script("arguments[0].click();", elem)
    time.sleep(duration * 0.8*play_rate)
    element = driver.find_element_by_xpath('// * // span[ @ title = "点赞（Q）"]')
    if element.get_attribute("class") == "like":
        driver.execute_script("arguments[0].click();", element)
    element = driver.find_element_by_xpath('// * // span[ @ title = "点赞（Q）"]/following-sibling::span[1]')
    if element.get_attribute("class") == "coin":
        try:
            driver.execute_script("arguments[0].click();", element)
            elem=driver.find_element_by_xpath('//*[@class="bi-btn" and text()="确定"]')
            driver.execute_script("arguments[0].click();", elem)
        except Exception as e:
            print(e)
            print('点击投币确认按钮失败，大概率是没登录')
    # 点广告
    try:
        elem = driver.find_element_by_xpath('// * [ @ class = "vcd"]')
        driver.execute_script("arguments[0].click();", elem)
    except Exception as e:
        print(e)
        print('点击广告失败')
    time.sleep(duration * 0.2*play_rate)
    return duration*play_rate

def join_act(driver,url='https://member.bilibili.com/platform/allowance/incomeCenter/pc?from=side_navigation'):
    try:
        driver.get(url)
        driver.find_element_by_xpath('//*[@class ="rec-more"]//span[text()="查看更多 "]').click()
        lis = driver.find_elements_by_css_selector('#app > div > div.content-page > div.home-page > div.content > div.punch-crad-list > div')
        for idx,li in enumerate(lis):
            try:
                elem=li.find_element_by_css_selector('#app > div > div.content-page > div.home-page > div.content > div.punch-crad-list > div:nth-child(%s) > div > div.item-btn'%(str(idx+1)))
                text=elem.get_attribute("textContent")
                if text.strip()=="去报名":
                    driver.execute_script("arguments[0].click();", elem)
                    # elem.click()
                else:
                    print(text)
            except Exception as e:
                print(e)
        print("自动参加活动成功")
    except Exception as e:
        print("自动参加活动失败")
        print(e)

def get_award(driver,url='https://member.bilibili.com/platform/allowance/upMission?task_id=20221205&task_type=2&href=index_web_msg'):
    try:
        driver.get(url)
        time.sleep(3)
        # elem=driver.find_element_by_xpath('//*[@class="desc" and text()="待领取"]/following-sibling::*[@class="coin-number"]')
        # jine=float(elem.get_attribute("textContent"))
        if 1:
            jiesu=driver.find_element_by_xpath('//*[@class="select-by-item"]//*[text()="已结束"]')
            ing=driver.find_element_by_xpath('//*[@class="select-by-item"]//*[text()="进行中"]')
            baokuang=driver.find_element_by_xpath('//*[@class="select-by-task"]//*[text()="爆款"]')
            fans=driver.find_element_by_xpath('//*[@class="select-by-task"]//*[text()="涨粉"]')
            for elem in [jiesu,ing,baokuang,fans]:
                isselect=elem.get_attribute("class")
                if isselect=="unselected":
                    driver.execute_script("arguments[0].click();", elem)

            lis=driver.find_elements_by_xpath('//*[@class="project-content"]/div')
            for idx,li in enumerate(lis):
                if idx ==len(lis)-1:break
                elem = li.find_element_by_xpath('//*[@class="project-content"]/div[%d]//*[contains(@class,"bcc-button get-challenge")]//span' % (idx + 2))
                date=li.find_element_by_xpath('//*[@class="project-content"]/div[%d]//div'%(idx + 2)).get_attribute('id')
                # elem=elem.find_element_by_xpath('//button[contains(@class,"bcc-button get-challenge")]//span')
                isward=elem.get_attribute("textContent")
                if isward=="待领奖":
                    driver.execute_script("arguments[0].click();", elem)
                    #todo 点击领奖之后还有一步 待验证
                    elem=li.find_element_by_xpath('//*[@id=%s]//*[text()="领奖并授权"]'%date)
                    driver.execute_script("arguments[0].click();", elem)
                    print('%s期领取奖励成功'%date)

                elif isward=="已领奖":
                    break
    except Exception as e:
        print(e)
        print('领取奖励失败')

def get_core_data(driver,ip,date,url='https://member.bilibili.com/platform/home'):
    try:
        record=ip+'\t'+date+'\n'
        driver.get(url)
        time.sleep(2)
        lis = driver.find_elements_by_css_selector('#cc-body > div.home-wrap.cc-content-body > div.data-card > div > div.section.video.clearfix > div.section-row.bcc-row.first > div')
        res=[]
        tt=['昨日新增粉丝数', '昨日视频播放数', '昨日评论数', '昨日弹幕数']
        for idx,li in enumerate(lis):
            # elem=li.find_element_by_css_selector('#cc-body > div.home-wrap.cc-content-body > div.data-card > div > div.section.video.clearfix > div.section-row.bcc-row.first > div:nth-child(%d) > div > div > div.value > span'%(idx+1))
            elem=li.find_element_by_css_selector('#cc-body > div.home-wrap.cc-content-body > div.data-card > div > div.section.video.clearfix > div.section-row.bcc-row.first > div:nth-child(%d) > div > div > div.data-card-top > div.diff > span'%(idx+1))
            text=elem.get_attribute("textContent")
            text=tt[idx]+': ' + text
            res.append(text)
        like_num=driver.find_element_by_css_selector('#cc-body > div.home-wrap.cc-content-body > div.data-card > div > div.section.video.clearfix > div:nth-child(2) > div:nth-child(1) > div > div > div.data-card-top > div.diff > span').get_attribute("textContent")
        res.append('昨日点赞数：'+like_num)
        #get 昨日收益
        shouyidate,shouyi,yearshouyi='xx','0.0','0.0'
        try:
            shouyidate=driver.find_element_by_xpath('//*[contains(text(),"日收益")]').get_attribute("textContent")
            shouyi=driver.find_element_by_xpath('//*[contains(text(),"日收益")]/following-sibling::p').get_attribute("textContent")
            yearshouyi=driver.find_element_by_xpath('//*[contains(text(),"近一年总收益")]/following-sibling::p').get_attribute("textContent")
        except Exception as e:
            print(e)
            print('%s:获取收益数据失败'%ip)
        res.append(shouyidate + '：' + shouyi)
        res.append('年收益：' + yearshouyi)
        with open('./core_data.txt','a') as f:
            f.write(record+'\t'.join(res)+'\n')
    except Exception as e:
        print(e)
        print('获取核心数据失败')

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
    date=datetime.datetime.now().strftime('%Y年%m月%d号 %H点%M分')
    print(date)
    file_nm='./videoB_url.txt'
    ip_lis = ['127.0.0.1:9225',"127.0.0.1:9222", '127.0.0.1:9223', '127.0.0.1:9224'] #gaoxiaozuqiu18750105941 wode dide 18305951310 huangde18829903397
    # ip_lis = ['127.0.0.1:9125',"127.0.0.1:9122", '127.0.0.1:9123', '127.0.0.1:9124'] #gaoxiaozuqiu wode dide huangde
    # minsheng2_huaji3=['127.0.0.1:9229','127.0.0.1:9228','127.0.0.1:9227','127.0.0.1:9226']#7965(足球), 7962（lol）,7963(三体),0739（怀旧）(1265need adentity)
    # ip_lis=minsheng2_huaji3+ip_lis
    if args.istest:
        ip_lis=['127.0.0.1:9223']
        file_nm='./videoB_url_test.txt'
    if args.issave:
        file = open(file_nm, 'w')
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
                for ip in ip_lis:
                    args.ip = ip
                    pid = get_pid(args)
                    if pid:
                        os.popen("taskkill /pid %s -t -f" % pid)
                    else:
                        print("%s 的进程没有杀死" % args.ip)
                print("下载失败退出")
                driver.quit()
                driver_service.stop()
                exit()
        file.close()
        print("下载成功退出")

    if args.isplay:
        print("start play...")
        try:
            with open(file_nm, 'r') as f:
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
            pid = get_pid(args)
            if pid:
                os.popen("taskkill /pid %s -t -f" % pid)
            else:
                print("%s 的进程没有杀死" % args.ip)
            print("自动播放失败退出")
            driver.quit()
            driver_service.stop()
            exit()
        driver.quit()
        driver_service.stop()
        pid = get_pid(args)
        if pid:
            os.popen("taskkill /pid %s -t -f" % pid)
        else:
            print("%s 的进程没有杀死" % args.ip)
        print("自动播放成功退出")

    if args.isgetdata:
        for ip in ip_lis:
            print('开始获取%s的数据'%ip)
            args.ip = ip
            try:
                driver, driver_service = init_driver(args)
                get_core_data(driver,ip,date)
                join_act(driver)
                get_award(driver)
                pid = get_pid(args)
                driver.quit()
                driver_service.stop()
                if pid:
                    os.popen("taskkill /pid %s -t -f" % pid)
                    print("%s的进程被杀死" % args.ip)
            except Exception as e:
                print(e)
                for ip in ip_lis:
                    args.ip = ip
                    pid = get_pid(args)
                    if pid:
                        os.popen("taskkill /pid %s -t -f" % pid)
                    else:
                        print("%s 的进程没有杀死" % args.ip)
                print("%s：获取核心数据失败退出"%ip)
                driver.quit()
                driver_service.stop()
                exit()
        print("获取核心数据成功退出")



if __name__ == '__main__':
    main(args)
