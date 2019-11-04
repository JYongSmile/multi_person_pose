@echo off
::set INTERVAL = 10
::timeout %INTERVAL%
::Again  

echo local_cap  
C:  
cd %~dp0
start python local_cap.py 
rem 使用ping命令暂停3s，这样可以看到调用python后的结果
::ping -n 10 127.0.0.1 > nul 
