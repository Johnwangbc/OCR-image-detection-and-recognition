OCR-image-detection-and-recognition 
MSER.py是目前的总工程，他会读取输入图像、规范尺寸、选择MSER通道、进行MSER、去除重叠和相交过大的区域、CNN模型筛选、去除包含关系中较小的区域、行归并。
CNN_win和CNN_linux是针对两个平台的文件，用于训练模型。MSER会调用它训练好的模型
model文件夹下的是模型
