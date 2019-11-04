
import cv2
import ast
import numpy as np
import os
import common
import time
import shutil
import PIL.Image as Image
import matplotlib.pyplot as plt
# 相对路径的import

from estimator import TfPoseEstimator
from networks import get_graph_path
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

def init_multi(target_size=(432, 368)):
    model_path = get_graph_path(model_name='cmu')
    model = TfPoseEstimator(model_path, target_size=target_size)
    return model

def process_multi(img_path, model):
    image = common.read_imgfile(img_path, None, None)
    filename = os.path.split(img_path)[1].split('.')[0]
    scales = ast.literal_eval(node_or_string='[None]')
    humans = model.inference(image, scales=scales)
    image = model.draw_humans(image, humans, imgcopy=False)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    path_save_2d = path = './results/' + filename + '_2d.jpg'
    cv2.imwrite(path_save_2d, image)
    # cv2.imshow('result', image)
    #cv2.waitKey()

    #poseLifting = Prob3dPose('/data/ai/JF/pose_estimation/multi_pose_estimator/lifting/models/prob_model_params.mat')
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

    num = len(pose_3d)
    if num % 2 == 0:
        l = num // 2
    else:
        l = num // 2 + 1

    IMAGES_PATH = './results/'
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    path_save_3d = './results/' + filename + '_3d.png'

    fig = plt.figure()
    for i, single_3d in enumerate(pose_3d):
        plot_pose(single_3d, i, l, fig, num)
    plt.savefig(path_save_3d)
    # plt.show()

    return path_save_2d, path_save_3d

img_path = './image/13.jpg'
model = init_multi()
t = time.time()
path_save_2d, path_save_3d = process_multi(img_path, model)
t = time.time() - t
print(t)