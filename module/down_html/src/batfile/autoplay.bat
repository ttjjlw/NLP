
cd D:\code\pycharm\project1\github\NLP\module\down_html\src && d:
rem taskkill /f /t /im chromedriver.exe
rem taskkill /f /t /im chrome.exe
rem python action_video.py --isheadless 1 --isplay 0 --issave 1 > log/saveurl.txt
start /B python action_video.py --ip 127.0.0.1:9121 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9121.txt
start /B python action_video.py --ip 127.0.0.1:9122 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9122.txt
start /B python action_video.py --ip 127.0.0.1:9123 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9123.txt
start /B python action_video.py --ip 127.0.0.1:9124 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9124.txt
start /B python action_video.py --ip 127.0.0.1:9125 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9125.txt
rem start /B python action_video.py --ip 127.0.0.1:9126 --isheadless 1 --isplay 1 --issave 0 --istest 0 > log/logplay9126.txt
rem start /B python action_video.py --ip 127.0.0.1:9127 --isheadless 1 --isplay 1 --issave 0 --istest 0 > log/logplay9127.txt
rem start /B python action_video.py --ip 127.0.0.1:9128 --isheadless 1 --isplay 1 --issave 0 --istest 0 > log/logplay9128.txt
rem start /B python action_video.py --ip 127.0.0.1:9129 --isheadless 1 --isplay 1 --issave 0 --istest 0 > log/logplay9129.txt
