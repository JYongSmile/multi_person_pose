import argparse
import ast
import logging
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import socket
import numpy
import sys

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
out_file = "./results/output_realtime_cmu.mp4"
# out_file_3d = "./result/output_m2_mobilenet_v2_large.mp4"









if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='multi-pose-estimation real_time')
    # parser.add_argument('--video', type=str, default='./video/single_person.mp4')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=1280x720')#432x368 720x1280
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    model = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    ##调用客户端摄像头
    # IP地址'0.0.0.0'为等待客户端连接
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
    print('connect from:' + str(addr))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(out_file, fourcc, 6, (480, 640))

    t = time.time()
    try:

        while 1:

            start = time.time()  # 用于计算帧率信息
            length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度

            stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
            data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
            print(image)

            width = int(image.shape[1])
            height = int(image.shape[0])


            #2D姿态识别
            humans = model.inference(image)
            if not args.showBG:
                image = np.zeros(image.shape)
            scales = ast.literal_eval(node_or_string='[None]')
            humans = model.inference(image, scales=scales)
            image = model.draw_humans(image, humans, imgcopy=False)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            # 生成2D姿态视频
            # video.write(image)
            #2D-->3D




            #计算帧率
            fps_time = time.time()
            # 将帧率信息回传，主要目的是测试可以双向通信
            end = time.time()
            seconds = end - start
            fps = 1 / seconds



            ##返回已处理图像到客户端
            # conn.send(bytes(str(int(fps)), encoding='utf-8'))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
            result, imgencode = cv2.imencode('.jpg', image, encode_param)
            # 建立矩阵
            data = numpy.array(imgencode)
            # 将numpy矩阵转换成字符形式，以便在网络中传输
            img_Data = data.tostring()

            # 先发送要发送的数据的长度
            # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
            conn.send(str.encode(str(len(img_Data)).ljust(16)))
            # print(img_Data)
            # 发送数据
            conn.send(img_Data)


            #######################
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("I'm done")
                break

            # k = cv2.waitKey(10) & 0xff
            # if k == 27:
            #     break
        video.release()
        sock.close()
        cv2.destroyAllWindows()


        t = time.time() - t
        print(t)
        # print('Tran_speed:', FrameNumber / t)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

logger.debug('finished+')


















