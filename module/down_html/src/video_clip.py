# 主要是需要moviepy这个库
from moviepy.editor import *
import os,random

video = VideoFileClip("有意思的视频/1.mp4")

# 剪辑视频，截取视频前0.1秒
start=random.random()+random.randint(0,10)
video_start = video.subclip(start,start+0.05)

start=random.random()+random.randint(0,10)
video_end = video.subclip(start,start+0.05)
# 拼接视频
final_clip = concatenate_videoclips([video_start,video,video_end])

final_clip.to_videofile("有意思的视频/target.mp4", fps=24, remove_temp=False,audio_codec='aac')




