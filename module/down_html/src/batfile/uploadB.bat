cd D:\code\pycharm\project1\github\NLP\ && d:
git pull origin main
rem 222爆笑 223足球 224名人 225搞笑足球
cd D:\code\pycharm\project1\github\NLP\module\down_html\src && d:
python db.py --isheadless 1 --ip 127.0.0.1:9222 --num 1 --video_addr "suiji" > log/log9222.txt
python db.py --isheadless 1 --ip 127.0.0.1:9223 --num 1 --video_addr "suiji"> log/log9223.txt
python db.py --isheadless 1 --ip 127.0.0.1:9224 --num 1 --video_addr "minren"> log/log9224.txt
python db.py --isheadless 1 --ip 127.0.0.1:9225 --num 1 --video_addr "minren"> log/log9225.txt
python db.py --isheadless 1 --ip 127.0.0.1:9222 --num 1 --video_addr "/tjl" >> log/log9222.txt
python db.py --isheadless 1 --ip 127.0.0.1:9223 --num 1 --video_addr "/tjl00">> log/log9223.txt
python db.py --isheadless 1 --ip 127.0.0.1:9224 --num 1 --video_addr "/tjl01">> log/log9224.txt
python db.py --isheadless 1 --ip 127.0.0.1:9225 --num 1 --video_addr "/tjl02">> log/log9225.txt
git add uploaded/ && git commit -m '已发布的视频' && git push origin main
rem python db.py --ip 127.0.0.1:9226 --num 2 --video_addr "huaijiu"> log/log9226.txt
rem python db.py --ip 127.0.0.1:9227 --num 2 --video_addr "santi"> log/log9227.txt
rem python db.py --ip 127.0.0.1:9228 --num 2 --video_addr "lol"> log/log9228.txt
rem python db.py --ip 127.0.0.1:9229 --num 2 --video_addr "zuqiu"> log/log9229.txt
