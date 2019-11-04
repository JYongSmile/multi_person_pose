
import socket
import cv2
import ast
import numpy as np
import numpy
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





target_size = (640, 480)
model_path = get_graph_path(model_name='cmu')
model = TfPoseEstimator(model_path, target_size=target_size)
scales = ast.literal_eval(node_or_string='[None]')



def process_multi():


    address = ('192.168.25.69', 8567)
    # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
    # socket.AF_INET：服务器之间网络通信
    # socket.SOCK_STREAM：流式socket , for TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 将套接字绑定到地址, 在AF_INET下,以元组（host,port）的形式表示地址.
    sock.bind(address)
    # 开始监听TCP传入连接。参数指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
    sock.listen(1)

    def recvall(sock, count):
        buf = b''  # buf是一个byte类型
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
        # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
        # 没有连接则等待有连接
    conn, addr = sock.accept()
    # print('connect from:' + str(addr))


    while 1:
        start = time.time()  # 用于计算帧率信息
        length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
        # print(int(length))
        stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
        data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像


        # width = int(image.shape[1])
        # height = int(image.shape[0])


        # 2D姿态识别

        humans = model.inference(image, scales=scales)
        image = model.draw_humans(image, humans, imgcopy=False)
        print(image)


        # 将帧率信息回传，主要目的是测试可以双向通信
        end = time.time()
        seconds = end - start
        fps = 1 / seconds



        # 返回已处理图像到客户端
        conn.send(bytes(str(int(fps)), encoding='utf-8'))

        # return image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print("I'm done")
            break;
    sock.close()
    cv2.destroyAllWindows()





    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # path_save_2d = path = './results/' + filename + '_2d.jpg'
    # cv2.imwrite(path_save_2d, image)
    # cv2.imshow('result', image)
    #cv2.waitKey()

    #poseLifting = Prob3dPose('/data/ai/JF/pose_estimation/multi_pose_estimator/lifting/models/prob_model_params.mat')
    # poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')
    # image_h, image_w = image.shape[:2]
    # standard_w = 640
    # standard_h = 480
    #
    # pose_2d_mpiis = []
    # visibilities = []
    # for human in humans:
    #     pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
    #     pose_2d_mpiis.append([(int(x * image_w + 0.5), int(y * image_h + 0.5)) for x, y in pose_2d_mpii])
    #     visibilities.append(visibility)
    #
    # pose_2d_mpiis = np.array(pose_2d_mpiis)
    # visibilities = np.array(visibilities)
    # transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
    # pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
    #
    # num = len(pose_3d)
    # if num % 2 == 0:
    #     l = num // 2
    # else:
    #     l = num // 2 + 1
    #
    # IMAGES_PATH = './results/'
    # if not os.path.exists(IMAGES_PATH):
    #     os.makedirs(IMAGES_PATH)
    # path_save_3d = './results/' + filename + '_3d.png'
    #
    # fig = plt.figure()
    # for i, single_3d in enumerate(pose_3d):
    #     plot_pose(single_3d, i, l, fig, num)
    # plt.savefig(path_save_3d)
    # # plt.show()
    #
    # return path_save_2d, path_save_3d

process_multi()