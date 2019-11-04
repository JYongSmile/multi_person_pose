import argparse
import ast
import logging
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import matplotlib.animation as animation

import common
# from tf_pose.estimator import TfPoseEstimator
from estimator import TfPoseEstimator
# from tf_pose.networks import get_graph_path, model_wh
from networks import get_graph_path,model_wh
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
out_file = "./results/output_1_cmu.mp4"
# out_file_3d = "./result/output_m2_mobilenet_v2_large.mp4"



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='multi-pose-estimation Video')
    parser.add_argument('--video', type=str, default='./video/1.mp4')
    parser.add_argument('--resolution', type=str, default='544x960', help='network input resolution. default=1280x720')#432x368 720x1280
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    model = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    filename = os.path.split(args.video)[1].split('.')[0]
    print('filename:',filename)
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    FrameNumber = cap.get(7)
    duration = FrameNumber / fps
    print('fps:',fps)
    print('duration:', duration)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    path_save_3d = './results/3D_video/1_3d.mp4'
    video_3d = cv2.VideoWriter(path_save_3d, fourcc, fps, (640, 480))

    t = time.time()
    while cap.isOpened() is True:
        ret, image = cap.read()
        if ret == True:
            humans = model.inference(image)
            if not args.showBG:
                image = np.zeros(image.shape)
            scales = ast.literal_eval(node_or_string='[None]')
            humans = model.inference(image, scales=scales)
            image = model.draw_humans(image, humans, imgcopy=False)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imshow('tf-pose-estimation result', image)
            video.write(image)
            fps_time = time.time()
            # ############## 2D -> 3D  ################
            poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')
            image_h, image_w = image.shape[:2]
            standard_w = 640
            standard_h = 480

            pose_2d_mpiis = []
            visibilities = []
            for human in humans:
                pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
                pose_2d_mpiis.append([(int(x * image_w + 0.5), int(y * image_h + 0.5)) for x, y in pose_2d_mpii])
                visibilities.append(visibility)

            pose_2d_mpiis = np.array(pose_2d_mpiis)
            visibilities = np.array(visibilities)
            transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
            pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
            # print(pose_3d,'\n')
            num = len(pose_3d)
            if num % 2 == 0:
                l = num // 2
            else:
                l = num // 2 + 1

            VIDEOS_PATH = './results/3D_video/'
            if not os.path.exists(VIDEOS_PATH):
                os.makedirs(VIDEOS_PATH)
            fig = plt.figure()
            for i, single_3d in enumerate(pose_3d):
                # print('i:',i)
                # print('num:', num)
                plot_pose(single_3d, i, l, fig, num)
            # plt.show()
            plt.savefig('./results/3D_video/img_3d.png')
            img_3d = cv2.imread('./results/3D_video/img_3d.png')
            video_3d.write(img_3d)

            # plt.show()

            #########################################
        if ret == False:
            break
        # if cv2.waitKey(1) == 27:
        #     break
    video_3d.release()
    video.release()
    cv2.destroyAllWindows()
    t = time.time() - t
    print(t)
    print('Tran_speed:',FrameNumber/t)
logger.debug('finished+')
