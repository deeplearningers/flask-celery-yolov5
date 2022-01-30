import requests
import random
import json

for i in range(10000):
    number = random.randint(1000,10000)
    p = {"messageId":str(number), "cameras":[{"cameraId":"0000","cameraRtsp":"rtsp://172.16.3.60/live.sdp"},
                                            {"cameraId":"1111","cameraRtsp":"rtsp://172.16.3.61/live.sdp"},
                                            {"cameraId":"2222","cameraRtsp":"rtsp://172.16.3.62/live.sdp"},
                                              {"cameraId":"3333","cameraRtsp":"rtsp://172.16.3.62/live.sdp"}]}
    #r = requests.post('http://172.16.1.97:5005/predict', params = p)
    payload = json.dumps(p)
    headers = {"Content-Type":"application/json","Cookie":"djiaalamcl-dkspmdamac488dda"}
    r= requests.request("POST","http://172.16.1.97:5005/predict",data = payload,headers = headers)
    r.content.decode("utf-8")
    if len(r.json()) >0 :
        print(r.json())