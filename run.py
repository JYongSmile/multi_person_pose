import argparse
import logging
import time
import ast
import os
import shutil
import cv2
import io
import numpy as np
import PIL.Image as Image
import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from matplotlib.backends.backend_agg import FigureCanvasAgg
# from bottle import run, get, HTTPResponse
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    # os.chdir('..')
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='image/12.jpg')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu/mobilenet_thin/mobilenet_v2_large/mobilenet_v2_small')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    filename = os.path.split(args.image)[1].split('.')[0]
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    t = time.time()
    #humans = e.inference(image, scales=scales)
    humans = e.inference(image, resize_to_default = (w > 0 and h > 0), upsample_size = args.resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    path = './results/' + filename + '_2d.jpg'
    #cv2.imwrite(path, image)
    cv2.imshow('tf-pose-estimation result', image)
    #cv2.waitKey()

    import matplotlib.pyplot as plt

    #fig = plt.figure()
    #a = fig.add_subplot(2, 2, 1)
    #a.set_title('Result')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(image)

    #bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    #bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    #a = fig.add_subplot(2, 2, 2)
    #plt.imshow(bgimg, alpha=0.5)
    #tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    #plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    #plt.colorbar()

    #tmp2 = e.pafMat.transpose((2, 0, 1))
    #tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    #tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    #a = fig.add_subplot(2, 2, 3)
    #a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    #plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    #plt.colorbar()

    #a = fig.add_subplot(2, 2, 4)
    #a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    #plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    #plt.colorbar()
#    plt.show()

    # import sys
    # sys.exit(0)

    logger.info('3d lifting initialization.')
    poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')

    image_h, image_w = image.shape[:2]
    standard_w = 640
    standard_h = 480

    pose_2d_mpiis = []
    visibilities = []
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)

    # print(pose_3d)

    # pose_3dqt = np.array(pose_3d[0]).transpose()
    # for point in pose_3dqt:
    # 	print(point)

    num = len(pose_3d)
    if num % 2 == 0:
        l = num // 2
    else:
        l = num // 2 + 1

    # IMAGES_PATH = './results/'
    # if not os.path.exists(IMAGES_PATH):
    #     os.makedirs(IMAGES_PATH)
    # IMAGES_FORMAT = ['.png', '.PNG']
    # IMAGE_SIZE = 600
    # IMAGE_ROW = 1
    # IMAGE_COLUMN = num
    # IMAGE_SAVE_PATH = './results/final.jpg'
    # to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))

    fig = plt.figure()
    for i, single_3d in enumerate(pose_3d):
        plot_pose(single_3d, i, l, fig, num)
    #path = './results/' + filename + '_3d.png'
    #plt.savefig(path)

    # 网页上显示图形
    # canvas = FigureCanvasAgg(fig)
    # buf = io.BytesIO()
    # canvas.print_png(buf)
    # data = buf.getvalue()
    # headers = {
    #     'Content-Type': 'image/png',
    #     'Content-Length': len(data)
    # }
    # @get('/')
    # def hello():
    #     return HTTPResponse(body=data, headers=headers)
    # run(port=8080)

    #     path = './results/person_' + str(i) + '.png'
    #     plt.savefig(path)
    #
    # image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
    #                os.path.splitext(name)[1] == item]
    # for y in range(1, IMAGE_ROW + 1):
    #     for x in range(1, IMAGE_COLUMN + 1):
    #         from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
    #             (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    #         to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    # shutil.rmtree(IMAGES_PATH)
    # os.mkdir(IMAGES_PATH)
    # to_image.save(IMAGE_SAVE_PATH)
    plt.show()


    pass
