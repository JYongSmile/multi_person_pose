import socket
import cv2
import numpy
import time
import sys
import threading
import json

send_flag = False
connect_flag = False
sock_h = None
frame_count = 0
 
def connect_f(ip, port):
    # 建立sock连接
    # address要连接的服务器IP地址和端口号
    #address = ('192.168.25.69', 8566)
	address = (ip,port,)
	
	while(True):
		global send_flag
		global connect_flag  # connnect_flag可以通过检测sock_h的状态来替换
		global sock_h
		if not connect_flag:
			try:
				# 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
				# socket.AF_INET：服务器之间网络通信
				# socket.SOCK_STREAM：流式socket , for TCP
				print("尝试建立连接")
				sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				# 开启连接
				sock.connect(address)
				connect_flag = True
				sock_h = sock
				print("成功建立连接")
			except socket.error as msg:
				print(msg)
				#sys.exit(1)
				time.sleep(1)
				connect_flag = False
				sock_h = None
	print("结束连接进程")
			
def SendVideo():
	# 建立图像读取对象
	capture = cv2.VideoCapture(0)
	# 读取一帧图像，读取成功:ret=1 frame=读取到的一帧图像；读取失败:ret=0
	ret, frame = capture.read()
	# 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
	# time.sleep()    
	global sock_h
	sock = sock_h
	global frame_count
	while(ret):
		if send_flag and sock_h:
			try:
				# 停止0.1S 防止发送过快服务的处理不过来，如果服务端的处理很多，那么应该加大这个值
				time.sleep(0.01)
				# cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
				# '.jpg'表示将图片按照jpg格式编码。
				result, imgencode = cv2.imencode('.jpg', frame, encode_param)
				# 建立矩阵
				data = numpy.array(imgencode)
				# 将numpy矩阵转换成字符形式，以便在网络中传输
				stringData = data.tostring()

				# 先发送要发送的数据的长度x
				# ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
				sock.send(str.encode(str(len(stringData)).ljust(16)))

				# 发送数据
				sock.send(stringData)
				print("发送第%d帧" % frame_count)
				frame_count += 1
				# 读取服务器返回值
				#receive = sock.recv(1024)
				#if len(receive): print(str(receive, encoding='utf-8'))
				
				# 读取下一帧图片
				ret, frame = capture.read()
				
			except socket.error as msg:
				print(msg)				
				global connect_f
				connect_flag = False		
		elif not sock_h:
			connect_flag = False		
		else:
			print("休眠中，等待发送指令......")
			time.sleep(1)
	print("结束发送进程")
	sock.close()
	#sys.exit(1)

def getFlags():
	while True:
		global sock_h
		global send_flag
		global connect_flag
		global frame_count
		sock = sock_h
		if not sock:
			print("连接还未建立")
			time.sleep(2)
		else:
			try:
				rev = sock.recv(1024)
				print("接收到数据")
				print(rev)
				rev_dic = json.loads(rev)
				print(rev_dic)
				#keys_l = ["send_f"]
				if "send_f" in rev_dic:
					send_flag = rev_dic["send_f"]
					if not send_flag:
						frame_count = 0
			except Exception as e:
				print(e)
				connect_flag = False
				time.sleep(2)
	print("结束接收进程")
	pass
	
noise suppression
if __name__ == '__main__':
    #SendVideo()
	#if cv2.waitKey(10) == 27:
	#	break
	thread_L = []
	address = ('192.168.25.69', 8566,)
	t1 = threading.Thread(target=connect_f, args=(address))
	t2 = threading.Thread(target=SendVideo)
	t3 = threading.Thread(target=getFlags)
	thread_L.append(t1)
	thread_L.append(t2)
	thread_L.append(t3)
	for t in thread_L:
		t.setDaemon(True)
		t.start()
		
	for t in thread_L:
		t.join()
	
	
