
cd D:\Code\pycharmf\project3\github\NLP\module\down_html\src && d:
rem taskkill /f /t /im chromedriver.exe
rem taskkill /f /t /im chrome.exe
python action_video.py --isheadless 1 --isplay 0 --issave 1 --istest 0 --isgetdata 0 > log/saveurl.txt
python batchB.py > log/batchB.txt
git add tmp/dealed_url.txt && git commit -m 'moviepyed' && git push origin main

