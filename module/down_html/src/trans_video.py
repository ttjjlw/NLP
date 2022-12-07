import cv2
import os

'''
批量视频抽帧
'''
def get_video_duration(video_path,filename):
    filename=os.path.join(video_path,filename)
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num =cap.get(7)
        duration = frame_num/rate
        return duration
    return -1

#读取数据集合
video_path = '/Users/tjl/job/pycharm/github/NLP/module/down_html/src/有意思的视频'

#获得文件名
start_dir_num = 30000000#从6位数开始编号
all_files = os.listdir(video_path)#获得所有的视频列表

#准备txt文件
f = open('D:\\video\my_dataset.txt','w')

for i in range(len(all_files)):
    video_type = all_files[i].split('.')[1]
    if video_type not in ['avi','MP4','flv','mov','mp4']:
        continue

    save_path = os.path.join(video_path+"save",str(start_dir_num))
    # 判断文件夹是否存在
    folder = os.path.exists(save_path)
    if not folder:
        os.makedirs(save_path)
        print("创建文件夹: ",str(start_dir_num))
    # subprocess.Popen('ffmpeg -i {} -r 6 -f image2 {}\%05d.jpg'.format(all_files[i],save_path),stdout=subprocess.PIPE)
    length=get_video_duration(video_path,all_files[i])
    zhen=int(35/length)

    cmd = 'ffmpeg -i {} -r {} -f image2 {}\%05d.jpg'.format(all_files[i],zhen,save_path)
    os.system(cmd)

    #将视频信息写入csv文件中 格式: 编号 帧数 类型
    frame_num = os.listdir(save_path)
    rowname = str(start_dir_num)+" "+str(len(frame_num))+" \n"
    f.write(rowname)
    start_dir_num+=1

f.close()
