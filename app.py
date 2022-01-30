from __future__ import absolute_import
import cv2
import torch
import random
import time
from flask import Flask,jsonify, request
import settings
from celery import Celery,group
from celery.exceptions import SoftTimeLimitExceeded
from utils.general import (check_img_size, non_max_suppression,
                           scale_coords, set_logging)
from utils.plots import plot_one_box
from utils.datasets import LoadWebcam
from names import namelist2
from torch.multiprocessing import set_start_method
from mythread import DataThread,PredictThread
import queue
import sys
sys.path.append('/home/')

app = Flask(__name__)
app.config.from_object(settings)


def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery

celery = make_celery(app)

CONFIG = {
    "weights": "weights/yolov5m.pt",
    "device": "cuda:1",
    "output": "/home/output",
    "imgsize": 640,
    "conf_thres": 0.6,
    "iou_thres": 0.7,
    "duration_time": 0.0,
}
def loadData(camera_rtsp,imgsz):
    dataset = LoadWebcam(camera_rtsp, img_size=imgsz)
    return dataset

def predictResult(dataset,message_id,camera_id,half,model,device,names,colors):
    result_tmp = []
    t0 = time.time()
    duration_time = 0
    object_count = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, CONFIG.get("conf_thres"), CONFIG.get("iou_thres"),
                                   classes=None, agnostic=False)
        p, s, im0 = path, '', im0s
        save_path = CONFIG.get("output") + "/" + message_id + '_' + camera_id +  '.jpg'
        if pred[0].shape[0] == 0:
            duration_time = duration_time + (time.time() - t0)
            if duration_time > CONFIG.get("duration_time"):
                #cv2.imwrite(save_path, im0s)
                print('no objects! (%.3fs %s)' % (duration_time, save_path))
                return result_tmp
            else:
                t0 = time.time()
                continue
        else:  # have objects
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det:
                        label1 = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s' % (names[int(cls)])
                        if names[int(cls)] in namelist2:
                            object_count = object_count + 1
                            plot_one_box(xyxy, im0, label=label1, color=colors[int(cls)], line_thickness=3)
                            data = {'messageId': message_id, 'cameraId': camera_id, 'name': label}
                            # 'coordinate': [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]}
                            #print(data)
                            result_tmp.append(data)
                # Stream results
            if object_count > 0:
                cv2.imwrite(save_path, im0)
                print('find attention objects! (%.3fs %s)' % ((time.time() - t0), save_path))
                return result_tmp
            else:
                now_time = time.time()
                duration_time = duration_time + (now_time - t0)
                if duration_time > CONFIG.get("duration_time"):
                    # cv2.imwrite(save_path, im0s)
                    print('no attention objects! (%.3fs %s)' % (duration_time, save_path))
                    return result_tmp
                else:
                    t0 = time.time()
                    continue
def predict_(message_id, camera_id, camera_rtsp,device,model,half,imgsz,names,colors):
    result_tmp = []
    if len(message_id) == 0 or len(camera_id) == 0 or len(camera_rtsp) == 0:
        return result_tmp
    #threads
    q=queue.Queue()
    thread_data = DataThread(q,func=loadData,args=(camera_rtsp,imgsz))
    thread_data.start()
    thread_data.join()
    thread_predict = PredictThread(func=predictResult,args=(q.get(),message_id,camera_id,half,model,device,names,colors))
    thread_predict.start()
    thread_predict.join()
    result_tmp = thread_predict.get_result()
    return result_tmp

@celery.task(name="tasks.first")
def predict_first(message_id, camera_id,camera_rtsp):
    result = []
    try:
        set_start_method('spawn',force=True)
        set_logging()
        device = CONFIG.get("device")
        half = True
        #Load model
        model = torch.load(CONFIG.get("weights"), map_location=device)['model']
        model.to(device).eval()
        imgsz = check_img_size(CONFIG.get("imgsize"), s=model.stride.max())  # check img_size
        if half:
            model.half()
        #names and colors
        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        result = predict_(message_id, camera_id, camera_rtsp,device,model,half,imgsz,names,colors)
    except SoftTimeLimitExceeded as te:
        print(te)
    except Exception as e:
        print(e)
    finally:
        return result

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    final ={}
    if request.method == 'POST':
        if request.content_type.startswith('application/json') and request.headers.environ['HTTP_COOKIE']==settings.SECRET_KEY:
            message_id = request.json.get('messageId')
            video_rtsp = request.json.get('cameras')
            numTasks = len(video_rtsp)
            if not (0 < numTasks <=4 and len(message_id) > 0):
                return jsonify(final)
            try:
                job = group([predict_first.s(message_id, video_rtsp[i]['cameraId'],
                                             video_rtsp[i]['cameraRtsp']) for i in range(0,numTasks)]).apply_async()
                res = job.get(timeout = 0.0)
                for i in range(len(res)):
                    if res[i]:
                        final['result'] = res[i]
                        break
            except Exception as e:
                print(e)
            finally:
                return jsonify(final)
        else:
            final['result'] = 'invalid request!'
            return jsonify(final)
    else:
        final['result'] = 'invalid request!'
        return jsonify(final)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, threaded=True,debug=False)
