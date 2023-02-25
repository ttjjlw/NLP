
cd D:\Code\pycharmf\project3\github\NLP\module\down_html\src && d:
set HOUR=%time:~0,2%
if %HOUR% EQU 19 (
    echo '' > log/logplay9121.txt 2>&1
    echo '' > log/logplay9122.txt 2>&1
    echo '' > log/logplay9123.txt 2>&1
    echo '' > log/logplay9124.txt 2>&1
    echo '' > log/logplay9125.txt 2>&1
    taskkill /f /t /im chromedriver.exe
    taskkill /f /t /im chrome.exe
)
rem taskkill /f /t /im chromedriver.exe
rem taskkill /f /t /im chrome.exe
rem python action_video.py --isheadless 1 --isplay 0 --issave 1 > log/saveurl.txt
start /B python action_video.py --ip 127.0.0.1:9121 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9121.txt 2>&1
start /B python action_video.py --ip 127.0.0.1:9122 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9122.txt 2>&1
start /B python action_video.py --ip 127.0.0.1:9123 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9123.txt 2>&1
start /B python action_video.py --ip 127.0.0.1:9124 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9124.txt 2>&1
start /B python action_video.py --ip 127.0.0.1:9125 --isheadless 1 --isplay 1 --issave 0 --istest 0 --isgetdata 0 >> log/logplay9125.txt 2>&1

