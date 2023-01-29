from moviepy.editor import *
import re

def download_video(tmp_dir,url='https://www.bilibili.com/video/BV1Z841137Rh/'):
    p=os.popen('you-get --format=dash-flv480 -o %s %s'%(tmp_dir,url))
    data=p.read()
    title=re.findall('title:\s+(.+)',data)
    print(title)
    return title[0]
def merge(tmp_dir,save_dir,title,url):
    title1='%s%s[00].mp4'%(tmp_dir,title)
    title2='%s%s[01].mp4'%(tmp_dir,title)
    video = VideoFileClip(title1)
    audio = AudioFileClip(title2)
    video_merge = video.set_audio(audio)
    video_merge.write_videofile("%s%s.mp4"%(save_dir,title),audio_codec='aac')
    with open(f'{tmp_dir}dealed_url.txt','w') as f:
        f.write(url+'\n')
    os.remove(f'{title1}')
    os.remove(f'{title2}')
    os.remove(f'{tmp_dir}{title}.cmt.xml')

if __name__ == '__main__':
    tmp_dir = './tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    save_dir = './tjl/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        with open(f'{tmp_dir}dealed_url.txt','r') as f:
            url_set=set(f.readlines())
    except:
        url_set={}

    title=download_video(tmp_dir)
    merge(tmp_dir,save_dir,title,url)