import uiautomation as auto
from time import sleep
import pandas as pd


auto.uiautomation.SetGlobalSearchTimeout(15)  # 设置全局搜索超时 60

def open_qqbox():
    """ 打开QQ群对话框界面
    :return:
    """
    # 1、任务栏窗口
    task_mainWindow = auto.PaneControl(searchDepth=1, Name='任务栏')
    # 2、通过任务栏窗口获取用户提示通知区域
    warn_part = task_mainWindow.ToolBarControl(Name="用户提示通知区域")
    # 3、获取并点击QQ按钮，找到【用户提示通知区域】中QQ的位置
    qq_button = warn_part.ButtonControl(foundIndex=1, searchDepth=1)
    qq_button.Click(waitTime=1.5)
    sleep(0.5)
    # 获取并激活QQ界面
    main_window = auto.WindowControl(searchDepth=1, Name='QQ')
    main_window.SetActive()
    sleep(0.5)
    # 获取搜索框
    search_Edit = main_window.EditControl(searchDepth=6, Name="搜索")
    sleep(0.5)
    # 搜索群名称
    search_Edit.SetFocus()
    search_Edit.SendKeys('QQ群名称')
    search_Edit.SendKeys('{Enter}')
    sleep(0.5)
    # 最大化打开的对话框
    dialog_box = auto.WindowControl(Name='QQ群名称', searchDepth=1)
    dialog_box.Maximize()


def save_content(LAST_MESS_TEMP):
    """ 复制聊天记录到指定文件中
    :return:
    """
    dialog_box = auto.WindowControl(Name='QQ群名称', searchDepth=1)
    dialog_box.SetActive()
    dialog_box.Maximize()
    message_win = dialog_box.ListControl(Name='消息', searchDepth=13)
    message_win.Click()
    auto.Click(800, 800)
    message_win.SendKeys('{Ctrl}A')
    message_win.SendKeys('{Ctrl}C')
    df = pd.read_clipboard(sep=r"\s+", encoding='utf-8', error_bad_lines=False)
    df.to_csv('message_tmp.txt', index=False, sep=' ', encoding='utf_8_sig')
    # 查找未写入文件的内容
    k = 0
    with open('message_tmp.txt', 'r', encoding='utf_8_sig') as fp:
        readlines = fp.readlines()
        for i, line in enumerate(readlines):
            if line == LAST_MESS_TEMP:
                k = i
                break
        LAST_MESS_TEMP = readlines[-1].strip()

    # 将未写入文件的内容写入文件
    with open('message.txt', 'a+', encoding='utf_8_sig') as fp:
        readlines = fp.readlines()
        for i in range(k, len(readlines)):
            fp.write(readlines[i])
    return LAST_MESS_TEMP

if __name__ == '__main__':
    open_qqbox()
    LAST_MESS_TEMP=''
    LAST_MESS_TEMP = save_content(LAST_MESS_TEMP)


