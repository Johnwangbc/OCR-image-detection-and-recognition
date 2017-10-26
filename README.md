OCR-image-detection-and-recognition。  
MSER.py是目前的总工程，他会读取输入图像、规范尺寸、选择MSER通道、进行MSER、去除重叠和相交过大的区域、CNN模型筛选、去除包含关系中较小的区域、行归并。目前这个文件只还没有加入识别的部分，正在训练调试中。 
CNN_linux用于训练二分类CNN模型。MSER会调用它训练好的模型。   
CNN_LSTM_CTC_linux是识别网络的模型。  
ImageDuplication是图像去重的，因为训练测试网络的样本会有很多都很相近（尤其是背景图），因此我用这个文件进行去重。  
model文件夹下的是模型。  
analyse_data是对训练资料进行分析，他会统计label的一些属性，并配有删除出现次数较小数据的函数。  
Generation Data文件夹是用于人工生成识别网络所需资料的工程。  
