from moviepy.editor import *
import re,os,random,shutil

def download_video(tmp_dir,url='https://www.bilibili.com/video/BV1Z841137Rh/'):
    p=os.popen('you-get --format=dash-flv480 -o %s %s'%(tmp_dir,url))
    data = p.buffer.read().decode(encoding='utf-8')
    # data=p.read(decode='utf-8')
    title=re.findall('title:\s+(.+)',data)
    try:
        print(title[0].strip())
    except:
        print(data)
    return title[0].strip()
def merge(tmp_dir,save_dir,title,url):
    title1='%s%s[00].mp4'%(tmp_dir,title)
    title2='%s%s[01].mp4'%(tmp_dir,title)
    if not os.path.exists("%s%s.mp4" % (save_dir, title)):
        video = VideoFileClip(title1)
        audio = AudioFileClip(title2)
        video_merge = video.set_audio(audio)
        video_merge.write_videofile("%s%s.mp4"%(save_dir,title),audio_codec='aac')
    if os.path.exists(f'{title1}'):os.remove(f'{title1}')
    if os.path.exists(f'{title2}'):os.remove(f'{title2}')
    if os.path.exists(f'{tmp_dir}{title}.cmt.xml'):os.remove(f'{tmp_dir}{title}.cmt.xml')

def video_clip(tmp_dir,save_dir,title):
    video = VideoFileClip("%s%s.mp4"%(tmp_dir,title))

    # 剪辑视频，截取视频前0.1秒
    start = random.random() + random.randint(0, 10)
    video_start = video.subclip(start, start + 0.05)

    start = random.random() + random.randint(0, 10)
    video_end = video.subclip(start, start + 0.1)
    # 拼接视频
    final_clip = concatenate_videoclips([video_start, video, video_end])
    new_title = re.sub(
        u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a#!~%&:;\(\)?,\"\.，《》：“”！？【】——。])", "", title)
    if not os.path.exists("%s%s.mp4" % (save_dir, new_title)):
        final_clip.to_videofile("%s%s.mp4"%(save_dir,new_title), fps=24, remove_temp=True, audio_codec='aac')
    with open(f'{tmp_dir}dealed_url.txt','a') as f:
        f.write(url+'\n')
    video.close()
    if os.path.exists("%s%s.mp4" % (tmp_dir, title)):os.remove("%s%s.mp4" % (tmp_dir, title))

def copy_file(save_dir,title):
    new_title  = re.sub(
        u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a#!~%&:;\(\)?,\"\.，《》：“”！？【】——。])", "", title)
    for i in range(6):
        dest_dir=save_dir[:-1]+'0%d'%i+save_dir[-1]
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        if not os.path.exists("%s%s.mp4" % (dest_dir, new_title)):
            shutil.copy("%s%s.mp4"%(save_dir,new_title),dest_dir)


if __name__ == '__main__':
    #根据tjlB_url.txt中的地址下载视频到tjl目录，曾经下载过的(由tmp/dealed_url.txt记载)不会再下载
    tmp_dir = './tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    save_dir = './tjl/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        with open(f'{tmp_dir}dealed_url.txt','r') as f:
            url_set=set([u.strip() for u in f.readlines()])
    except:
        url_set={}
    with open('./tjlB_url.txt','r') as f:
        lines=f.readlines()
    for url in lines:
        url=url.strip()
        if url in url_set:continue
        title=download_video(tmp_dir,url=url)
        title = title.replace('/', '-')
        merge(tmp_dir,tmp_dir,title,url)
        video_clip(tmp_dir,save_dir,title)
        copy_file(save_dir,title)