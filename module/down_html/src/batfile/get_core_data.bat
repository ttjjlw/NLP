
cd D:\code\pycharm\project1\github\NLP\module\down_html\src && d:
rem taskkill /f /t /im chromedriver.exe
rem taskkill /f /t /im chrome.exe
python action_video.py --isheadless 1 --isplay 0 --issave 0 --istest 0 --isgetdata 1 > log/get_core_data.txt
python send_email.py