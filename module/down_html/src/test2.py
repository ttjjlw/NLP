from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler


def task():
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    print(ts)


def task2():
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    print(ts + '666!')


def func():
    # 创建调度器BlockingScheduler()
    scheduler = BlockingScheduler()
    scheduler.add_job(task, 'interval', seconds=3, id='test_job1')
    # 添加任务，时间间隔为5秒
    scheduler.add_job(task2, 'interval', seconds=5, id='test_job2')
    scheduler.start()


func()