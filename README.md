OCR-image-detection-and-recognition。  
OCR.py是目前的总工程，他会读取输入图像、规范尺寸、选择MSER通道、进行MSER、去除重叠和相交过大的区域、CNN模型筛选、去除包含关系中较小的区域、行归并、识别。   
CNN_linux用于训练二分类CNN模型。MSER会调用它训练好的模型。   
CNN_LSTM_CTC_linux是识别网络的模型，他会先读取数据，制作h5文件，之后进行训练。  
ImageDuplication是图像去重的，因为训练测试网络的样本会有很多都很相近（尤其是背景图），因此我用这个文件进行去重。  
model文件夹下的是模型。  
analyse_data是对训练资料进行分析，他会统计label的一些属性并写入txt。还可以对数据进行增强和筛选。
Generation Data文件夹是用于人工生成识别网络所需资料的工程。  
