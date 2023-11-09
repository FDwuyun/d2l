# coding=utf-8
import logging
import requests
import json
import schedule
import time
import threading
from datetime import datetime, timedelta


def main():
    print("1")

schedule.every().day.at("20:10:30").do(main)  # 设置每天的执行时间
schedule.every().day.at("20:10:40").do(main)  # 设置每天的执行时间
schedule.every().day.at("20:10:50").do(main)  # 设置每天的执行时间
schedule.every().day.at("20:11").do(main)  # 设置每天的执行时间

while True:
    schedule.run_pending()
    time.sleep(1)