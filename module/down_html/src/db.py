import selenium
from selenium import webdriver
import pathlib
import time
import shutil,os
from selenium.webdriver.common.keys import Keys

import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--video_addr',type=str,default=r"D:\code\pycharm\project1\github\video1")
parser.add_argument('--video_label',type=str,default='label1,label2')
parser.add_argument('--ip',type=str,default='127.0.0.1:9222')
parser.add_argument('--video_describe',type=str,default='视频')

args,_=parser.parse_known_args()

args.video_addr=r"D:\code\pycharm\project1\github\space"
args.video_label="科技，宇宙，太空，未来"
args.video_describe="震撼人心的视频"

move_dir=args.video_addr+"_move"

if not os.path.exists(move_dir):
    os.makedirs(move_dir)
def publish_bilibili(driver,path_mp4):
    '''
     作用：发布b站视频
    '''

    # 进入创作者页面，并上传视频
    # driver.refresh()
    driver.get("https://member.bilibili.com/platform/upload/video/frame")

    try:
        alert = driver.switch_to.alert()
        print('alert text:', alert.text)
        alert.accept()
    except:
        pass
    time.sleep(3)
    driver.switch_to.frame(driver.find_element_by_xpath('//iframe[@name="videoUpload"]'))
    print(path_mp4)
    driver.find_element_by_xpath('//input[@type="file" and contains(@accept,"mp4")]').send_keys(path_mp4)

    # 等待视频上传完成
    while True:
        try:
            driver.find_element_by_xpath('//*[text()="上传完成"]')
            break;
        except Exception as e:
            print("视频还在上传中···")

    print("视频已上传完成！")

    # # 添加封面
    driver.find_element_by_xpath('//*[text()="更改封面"]').click()
    driver.find_element_by_xpath('//*[@class="cover-cut"]//*[contains(text(),"完成")]').click()
    # # time.sleep(1)
    # driver.find_element_by_xpath('//div[text()="图片上传"]').click()
    # # time.sleep(1)
    # driver.find_element_by_xpath('//input[@type="file"]').send_keys(path_cover)
    # time.sleep(3)
    # driver.find_element_by_xpath('//*[text()="确定"]').click()

    # 输入标题
    # driver.find_element_by_xpath('//input[contains(@placeholder,"标题")]').clear()
    # driver.find_element_by_xpath('//input[contains(@placeholder,"标题")]').send_keys(describe[:describe.index(" #")])

    # 选择分类
    element = driver.find_element_by_xpath('//*[contains(@class,"select-container")]')
    print(element.get_attribute("class"))

    element.click()
    time.sleep(1)
    driver.find_element_by_xpath('//*[@class="f-item-content" and text()="知识"]').click()
    driver.find_element_by_xpath('//*[@class="item-main" and text()="科学科普"]').click()

    # 选择标签
    time.sleep(2)
    driver.find_element_by_xpath(
        '//*[text()="参与话题："]/..//*[@class="tag-topic-list"]/span[1]//*[@class="hot-tag-item"]').click()
    for label in args.video_label.strip().split('，'):
        print(label)
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
    driver.find_element_by_xpath('//*[@class="submit-add" and text()="立即投稿"]').click()

def main(args):
    # 基本信息
    # 视频存放路径
    catalog_mp4 = args.video_addr
    # 视频描述
    # time.sleep(10)
    options = webdriver.ChromeOptions()
    options.add_experimental_option("debuggerAddress", args.ip)
    driver = webdriver.Chrome(executable_path='.././chromedriver',options=options) #
    # driver.maximize_window()

    driver.implicitly_wait(10)

    path = pathlib.Path(catalog_mp4)

    # 视频地址获取
    path_mp4 = ""
    idx=0
    for i in path.iterdir():
        if (".mp4" in str(i)):
            path_mp4 = str(i)
        print("检查到视频路径：" + path_mp4)
        # publish_bilibili(driver, path_mp4)
        try:
            publish_bilibili(driver, path_mp4)
            shutil.move(path_mp4, move_dir)
            idx+=1
            if idx>4:break
        except:
            time.sleep(1)
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




    # 开始执行b站视频发布

if __name__ == '__main__':
    main(args)