import cv2
import numpy as np

from alerts.ManageAlerts import ManageAlerts
from config import *
from EngineApp.EngineApp import EngineApp
from equipment.ManageEquipment import ManageEquipment
from tools.preparePrediction import define_work_region, classesToDetect
from tools.image_processing import ImageProcessing
from tools.multithreading import camThread


def inference(department, shift, id_camera, objects=None, videos=None):
    if videos != None and len(videos) > 0 and len(videos) == len(id_camera):
        videos = ['dataset/' + s for s in videos]
    else:
        # TODO
        captures = cv2.VideoCapture('rtsp://username:password@IP/PORT')
    
    threads = []
    try:
        for i in range(len(id_camera)):
            t = camThread(id_camera[i], department, shift, objects[i], videos[i])
            t.start()
            threads.append(t)
        
        for thread in threads:
            print(f'JOIN THREAD {thread} | {thread.is_alive()} | {thread.getName()}')
            thread.join()
            print(f'THREAD {thread} JOINED | {thread.is_alive()} | {thread.getName()}')

    except ValueError as err:
        print(f'Excpetion: {err.args}')


def inferenceInVideo(department, shift, id_camera, classes_ids, classes_names, video):
    engine = EngineApp()
    alertObj = ManageAlerts()

    experiment, model = engine.defineModelParameters(model_id=2)

    mask_points = np.array([[930, 1], [1236, 1], [1236, 425]])
    show = False

    video = f'dataset/{video}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(f'{video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps:", fps)
    print("w:", w)
    print("h:", h)
    result = cv2.VideoWriter('outputs/testMP4.mp4', fourcc, fps, (w, h))

    frames = []
    status_alert = [False] * len(classes_ids)
    count_alert = [0] * len(classes_ids)
    stop_alert = [0] * len(classes_ids)
    cls_conf = 0.10
    cont_frame = -1
    X = True

    while cap.isOpened():
        success, frame = cap.read()
        cont_frame += 1

        if not success:
            break

        if success and cont_frame % 1 == 0:
            print(f'Reading frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} of {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')

            copy_frame = frame.copy()
            ip = ImageProcessing(copy_frame)

            # Define work region
            x_upper, y_upper = 1, 370  # y = 402
            x_bottom, y_bottom = 1239, 905
            bbox_work_region = [1, 370, 1239, 905]
            work_region = ip.cutImage(copy_frame, x_upper, y_upper, x_bottom, y_bottom, show)
            # cv2.imwrite('dataset/work_region.jpg', work_region)

            # Apply mask in the work region
            #mask_work_region = ip.drawMask(work_region, mask_points, True)

            # Detecting objects
            img_detections, boxes_detections, cls_detections, scores_detections = engine.objectDetection(experiment, model, work_region, cls_conf, classes_ids)
            # Converting tensors to list
            boxes_detections = boxes_detections.tolist()
            for i in range(len(boxes_detections)):
                boxes_detections[i] = list(map(int, boxes_detections[i]))
            cls_detections = list(map(int, cls_detections))
            scores_detections = scores_detections.tolist()

            # Replace work region image with detections into original image
            detections_work_region = engine.replaceWorkRegionOnOriginalImage(frame, img_detections, bbox_work_region)
            # cv2.imshow('Original Image with Object Dectected', detections_work_region)
            # if cv2.waitKey(1) == 27:
            #     cv2.destroyAllWindows()
            #     break

            # Alert managment
            counter = alertObj.countEachClass(classes_ids, cls_detections, scores_detections, cls_conf)
  
            # output: [4, 4, 0] -> Pessoa, Capacete, Colete

            if X:
                frames, counter_classes_prev, dt_inicio = \
                    alertObj.alertInspector(
                        X,
                        counter,
                        shift,
                        id_camera,
                        department,
                        classes_names,
                        detections_work_region,
                        status_alert,
                        count_alert,
                        stop_alert,
                        frames
                    )
                X = False
            else: 
                frames, counter_classes_prev, _ = \
                    alertObj.alertInspector(
                        X,
                        counter,
                        shift,
                        id_camera,
                        department,
                        classes_names,
                        detections_work_region,
                        status_alert,
                        count_alert,
                        stop_alert,
                        frames,
                        counter_classes_prev,
                        dt_inicio
                    )

            cv2.imshow("Detections", detections_work_region)
            # result.write(detections_work_region)
            cv2.waitKey(0)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
    cap.release()
    result.release()

def inferenceAux(classes_ids, video):
    engine = EngineApp()

    experiment, model = engine.defineModelParameters(model_id=1)

    video = f'dataset/{video}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(f'{video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result = cv2.VideoWriter('outputs/mask_test.mp4', fourcc, fps, (w, h))

    frames = []
    cls_conf = 0.10
    cont_frame = -1


    while cap.isOpened():
        success, frame = cap.read()
        cont_frame += 1

        if not success:
            break

        if success and cont_frame % 1 == 0:
  
            img_detections, boxes_detections, cls_detections, scores_detections = engine.objectDetection(experiment, model, frame, cls_conf, classes_ids)


            cv2.imshow("Detections", img_detections)
            result.write(img_detections)
    
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
    cap.release()
    result.release()

if __name__ == "__main__":
    equipManagment = ManageEquipment()
    models = equipManagment.defineOperation()

    department = "Montagem"
    shift = "Noite"

    # work_region = define_work_region(image='dataset/frame100.jpg', show=True)
    # work_region_points = np.array([[123, 403], [925, 390], [1215, 810], [934, 870], [642, 899], [318, 889], [7, 837]])

    video_cam1 = 'thread_test_3s.mp4'
    video_cam2 = 'thread_test_4s.mp4'
    # video_cam1 = 'alert_check.mp4'
    # video_cam2 = 'alert_check_2.mp4'
    video_cam3 = 'test_mask.mp4'
    #video_4 = 'video_4.mp4'

    '''cameras, objects, objects_to_detect = [], [], []
    for model in models:
        cameras.append(model['id_cam'])
        if model['job'] == 'OD':
            objects.append(model['objects'])

    inference(
        department=department,
        shift=shift,
        id_camera=cameras,
        objects=objects,
        videos=[video_cam1, video_cam2]
    )'''

    objects_ids, objects_names = classesToDetect(models)
    #inferenceInVideo(department, shift, '10.0.0.1', objects_ids[0], objects_names[0], video_cam3)
    inferenceAux(objects_ids[0], video_cam3)