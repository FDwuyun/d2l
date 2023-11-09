# coding=utf-8
import logging
import requests
import json
import schedule
import time
import threading
from datetime import datetime, timedelta

headers = {
    'x-id-token': 'eyJhbGciOiJSUzUxMiJ9.eyJBVFRSX3VzZXJObyI6IjIwMjMyMDI3MjAiLCJzdWIiOiIyMDIzMjAyNzIwIiwiaXNzIjoidWlzLm53cHUuZWR1LmNuIiwiZGV2aWNlSWQiOiJaTGRZZGRDbHM3Y0RBRitwNnJOUDF0M3ciLCJBVFRSX2lkZW50aXR5VHlwZUlkIjoiUzAyIiwiQVRUUl9hY2NvdW50SWQiOiI0M2JmNjYxMDI2MjAxMWVlNDdhNDA3MGYyOTZmZDgzNSIsIkFUVFJfdXNlcklkIjoiNDM5NDM3NjAyNjIwMTFlZTQ3YTQwNzBmMjk2ZmQ4MzUiLCJBVFRSX2lkZW50aXR5VHlwZUNvZGUiOiJTMDIiLCJBVFRSX2lkZW50aXR5VHlwZU5hbWUiOiLnoJTnqbbnlJ8iLCJBVFRSX29yZ2FuaXphdGlvbk5hbWUiOiLorqHnrpfmnLrlrabpmaIiLCJBVFRSX3VzZXJOYW1lIjoi546L6ICA5aKeIiwiZXhwIjoxNjk5NjIwNDkwLCJBVFRSX29yZ2FuaXphdGlvbklkIjoiMDYxMDAiLCJpYXQiOjE2OTcwMjg0OTAsImp0aSI6IklkLVRva2VuLUtGa0lpVlRJOGN1SWxSNEkiLCJyZXEiOiJjb20ubGFudHUuTW9iaWxlQ2FtcHVzLm53cHUiLCJBVFRSX29yZ2FuaXphdGlvbkNvZGUiOiIwNjEwMCJ9.KCmsD3K7l_GqYwV8Ce_touol4ArecpPO1Qi38iWtbS5gpVZs98MzBi8wC7IG4QOQ06Owm56qkeDvc8GVKNQJT5n4Hc3dpLiuGRTttXCDxWsCZAn85bc4CFMqM7Lk8WrTDFaQ1j_6Gmgp0RAe7f-Bg1kl8ZOVADnZt0W8E10DXX0',
    'roomtypeids': 'undefined',
    'user-agent': 'Mozilla/5.0 (Linux; Android 12; ANN-AN00 Build/HONORANN-AN00; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/92.0.4515.105 Mobile Safari/537.36 SuperApp',
    'content-type': 'application/json;charset=UTF-8',
    'accept': '*/*',
    'origin': 'https://art-reservation.nwpu.edu.cn',
    'x-requested-with': 'com.lantu.MobileCampus.nwpu',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'referer': 'https://art-reservation.nwpu.edu.cn/h5/pages/apply/reserve',
    'accept-encoding': 'gzip, deflate',
    'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'cookie': 'userToken=eyJhbGciOiJSUzUxMiJ9.eyJBVFRSX3VzZXJObyI6IjIwMjMyMDI7f-Bg1kl8ZOVADnZt0W8E10DXX0; Domain=.nwpu.edu.cn; Path=/'
}
success_thread_found = threading.Event()

def send_post_request(room_id, rule_id):
# 请求头
    current_date = datetime.now()
    # 修改1
    next_date = current_date + timedelta(days=3)
    formatted_date = next_date.strftime('%Y-%m-%d')
    # 修改2
    begin_time = "21:00"
    end_time = "22:30"
    # if next_date.weekday() < 4:  # 周一到周四
    #     begin_time = "21:00"
    #     end_time = "22:30"
    # else:  # 其他日期
    #     begin_time = "21:30"
    #     end_time = "23:00"
# 请求体
    data = {
        "applyExtendList": [
            {
                "applyDate": formatted_date,
                "beginTime": begin_time,
                "endTime": end_time,
                "seatNumber": ""
            }
        ],
        "isCycle": 0,
        "applicant": "06100",
        "applicantLabel": "计算机学院",
        "docUrl": [],
        "leaderId": "43bf6610262011ee47a4070f296fd835",
        "leaderName": "王耀增",
        "leaderNo": "2023202720",
        "phone": "18545016760",
        "remark": "",
        "roomId": room_id,
        "subject": "羽毛球",
        "allowAgentRa": 0,
        "participant": [
            "c8360b80262311ee47a4070f296fd835"
        ],
        "useRuleId": rule_id,
        "seatCount": "2",
        "actualUserId": "",
        "actualUserAccountName": "",
        "actualUserName": ""
    }

    # 发送POST请求
    url = 'https://art-reservation.nwpu.edu.cn/api/reservation'
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # 打印响应内容
    print(response.status_code)
    print(response.text)
    if response.status_code == 200:
        print(f"Room {room_id} successfully reserved.")
        success_thread_found.set()
        current_time = time.strftime("%H:%M:%S")
        # log.debug("current time"+current_time)


def main():
# # 列出要发送请求的roomId列表,友谊西苑测试样本
    # 修改4
    # roomIds = ["1537703257741627394", "1537703741277769729","1537704046958645249","1537704454527553537","1537704781515493377"]  # 替换为实际的roomId列表
    # ruleIds = ["1612636086397620225", "1612636169897824257","1612636329566588930","1612636419299528705","1612636477017346049"]  # 替换为实际的roomId列表
    # 长安翱翔
    roomIds = ["1635184009838305282","1635470337972871170","1635472352417390593","1635472862390231041","1635478563590217729","1635528498090016770","1635530193079566337","1635533169282187266"]
    ruleIds = ["1635185784762900481","1635624118733701121","1635624538864549889","1635624927798165506","1635625905670782978","1635626194108874753","1635626513651924993","1635627740787834881"]

    # 创建线程并启动
    threads = []
    for room_id, rule_id in zip(roomIds, ruleIds):
        thread = threading.Thread(target=send_post_request, args=(room_id, rule_id))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成或成功线程找到
    for thread in threads:
        thread.join()

    # 如果成功线程找到，停止其他线程
    if success_thread_found.is_set():
        # log.debug("A successful reservation was found. Stopping other threads.")
        for thread in threads:
            thread.join()

# 修改3
schedule.every().day.at("22:59:20").do(main)  # 设置每天的执行时间
schedule.every().day.at("22:59:30").do(main)  # 设置每天的执行时间
schedule.every().day.at("22:59:40").do(main)  # 设置每天的执行时间
schedule.every().day.at("22:59:50").do(main)  # 设置每天的执行时间
schedule.every().day.at("23:00").do(main)  # 设置每天的执行时间

# log = logging.getLogger()
# # 运行定时任务
# log.debug("Program service applying")
while True:
    schedule.run_pending()
    time.sleep(1)