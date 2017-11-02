# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py
import  cv2

# def deletepartdata(class_label,class_num_label,data,label,seqlength):
#     # 找出标签中出现次数小于阈值的文字，把包含这些文字的数据从data,label,seqlength中去除
#     delete_list = []
#     print('处理delete_list')
#     for i in range(len(class_num_label)):
#         if class_num_label[i] < 8000:
#             delete_list.append(class_label[i])
#
#     # 以下操作是删除delete_list中的数据
#     data = list(data)
#     label = list(label)
#     seqlength = list(seqlength)
#     i = 0
#     j = 0
#     while i <= len(label)-1:
#         print ('正在删除delete_list中的数据，处理到第%d个' % i)
#         while j <= len(label[i])-1:
#             if label[i][j] in delete_list:
#                 j = 0
#                 data.pop(i)
#                 label.pop(i)
#                 seqlength.pop(i)
#                 break
#             else:
#                 j += 1
#         i += 1
#         j = 0
#
#     # 以下操作是统计有哪些维度出现过
#     label_index = set()
#     for i in range(len(label)):
#         for j in range(len(label[i])):
#             if label[i][j] != -1:
#                 label_index.add(label[i][j])
#             else:
#                 break
#     label_index = list(label_index)
#
#     # 以下操作是制作新的文字列表alphabet_final
#     with open('/home/wangbc1/OCR/alphabet_final.txt', 'r') as f:
#         alphabet = f.readline().decode('utf8')
#         alphabet = list(alphabet)
#     with open('/home/wangbc1/OCR/alphabet_final.txt', 'w') as f:
#         for i in label_index:
#             f.write(alphabet[i].encode('utf8'))
#     # 以下操作是重设label值
#     for i in range(len(label)):
#         print('正在重设label数值，处理到第%d个' % i)
#         for j in range(len(label[i])):
#             if label[i][j] != -1:
#                 label[i][j] = label_index.index(label[i][j])
#             else:
#                 break
#
#     return np.asarray(data,np.float32),np.asarray(label,np.int32),np.asarray(seqlength,np.int32)
#
# def dataaugmentation(class_label,class_num_label,data,label,seqlength):
#     # 进行数据增强，包括加噪声、平移、旋转
#     low_list = []
#     print('处理low_list')
#     for i in range(len(class_num_label)):
#         if class_num_label[i] < 238:
#             low_list.append(class_label[i])
#
#     data = list(data)
#     label = list(label)
#     seqlength = list(seqlength)
#     rows = len(label)
#     cols = len(label[0])
#     for i in range(rows):
#         for j in range(cols):
#             if label[i][j] in low_list:
#                 for k in range(10):
#                     data.append(gaussiannoise(data[i]))
#                     label.append(label[i])
#                     seqlength.append(24)
#                     H = np.float32([[1, 0, np.random.randint(-15,15)], [0, 1, np.random.randint(-15,15)]])
#                     M = cv2.getRotationMatrix2D((data[i].shape[1] / 2, data[i].shape[0] / 2), np.random.randint(-10,10), 1)
#                     data.append(cv2.warpAffine(data[i], H, (data[i].shape[1], data[i].shape[0]),
#                                                     borderMode=cv2.INTER_LINEAR))
#                     label.append(label[i])
#                     seqlength.append(24)
#                     data.append(cv2.warpAffine(data[i], M, (data[i].shape[1], data[i].shape[0]),
#                                                     borderMode=cv2.INTER_LINEAR))
#                     label.append(label[i])
#                     seqlength.append(24)
#                 break
#             else:
#                 continue
#     return np.asarray(data,np.float32),np.asarray(label,np.int32),np.asarray(seqlength,np.int32)
#
# def gaussiannoise(img):
#     # 为彩色图像加入高斯噪声
#     grayscale = 256
#     w = img.shape[1]
#     h = img.shape[0]
#     newimg = np.zeros((h, w, 3), np.uint8)
#     param = 30
#
#     for x in range(0, h):
#         for y in range(0, w, 2):
#             r1 = np.random.random_sample()
#             r2 = np.random.random_sample()
#             z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
#             z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
#
#             fxy_0 = int(img[x, y, 0] + z1)
#             fxy_1 = int(img[x, y, 1] + z1)
#             fxy_2 = int(img[x, y, 2] + z1)
#             fxy1_0 = int(img[x, y + 1, 0] + z2)
#             fxy1_1 = int(img[x, y + 1, 1] + z2)
#             fxy1_2 = int(img[x, y + 1, 2] + z2)
#             # 0
#             if fxy_0 < 0:
#                 fxy_val_0 = 0
#             elif fxy_0 > grayscale - 1:
#                 fxy_val_0 = grayscale - 1
#             else:
#                 fxy_val_0 = fxy_0
#             if fxy1_0 < 0:
#                 fxy1_val_0 = 0
#             elif fxy1_0 > grayscale - 1:
#                 fxy1_val_0 = grayscale - 1
#             else:
#                 fxy1_val_0 = fxy1_0
#             # 1
#             if fxy_1 < 0:
#                 fxy_val_1 = 0
#             elif fxy_1 > grayscale - 1:
#                 fxy_val_1 = grayscale - 1
#             else:
#                 fxy_val_1 = fxy_1
#             if fxy1_1 < 0:
#                 fxy1_val_1 = 0
#             elif fxy1_1 > grayscale - 1:
#                 fxy1_val_1 = grayscale - 1
#             else:
#                 fxy1_val_1 = fxy1_1
#             # 2
#             if fxy_2 < 0:
#                 fxy_val_2 = 0
#             elif fxy_2 > grayscale - 1:
#                 fxy_val_2 = grayscale - 1
#             else:
#                 fxy_val_2 = fxy_2
#             if fxy1_2 < 0:
#                 fxy1_val_2 = 0
#             elif fxy1_2 > grayscale - 1:
#                 fxy1_val_2 = grayscale - 1
#             else:
#                 fxy1_val_2 = fxy1_2
#
#             newimg[x, y, 0] = fxy_val_0
#             newimg[x, y, 1] = fxy_val_1
#             newimg[x, y, 2] = fxy_val_2
#             newimg[x, y + 1, 0] = fxy1_val_0
#             newimg[x, y + 1, 1] = fxy1_val_1
#             newimg[x, y + 1, 2] = fxy1_val_2
#     return newimg
#-------以上为npy版本的数据增强和删除相关函数，由于数据太大，已经将数据保存为h5文件，所以停止以上代码的维护------
#-------如果你仍然使用npy，以上代码就可以用，不过经过试验表明，数据增强效果不如直接增加数据量--------------------

def analyselabel(label):
    # 计算label的总数据量，总类别数，最大类别数，最小类别数，平均类别数，平均长度
    mean_len = 0.0
    class_label = []
    class_num_label = []

    print('正在统计......')
    data_num = len(label)
    for i in range(data_num):
        each_row = list(label[i])
        try:
            mean_len += each_row.index(-1)
        except:
            mean_len += label.shape[1]
        for j in each_row:
            if j != -1:
                if j in class_label:
                    class_num_label[class_label.index(j)] += 1
                else:
                    class_label.append(j)
                    class_num_label.append(1)
            else:
                break
    mean_len = mean_len / data_num
    data_class = len(class_label)
    max_class_num = max(class_num_label)
    min_class_num = min(class_num_label)
    mean_class_num = sum(class_num_label) / len(class_num_label)
    return class_label,class_num_label,data_num,data_class,max_class_num,min_class_num,mean_class_num,mean_len

def maxlinelength(path):
    # 以下是统计样本一行最多多少字
    max_len = 0
    with open(path + '/label.txt', 'r',encoding='utf-8') as f:
        a = f.readlines()
        for i in range(len(a)):
            b = a[i].split()[1]
            max_len = max(max_len,len(b))
    return max_len

def maxclass(path):
    max_num = 0
    for k in range(6):
        f2 = h5py.File(path + '/data_final_%d.h5' % (k + 1), 'r')
        label = f2['label'][:]
        max_num = max(max_num, np.max(label))
    print(max_num)

def drawplot(class_num_label_final):
    # 将linux统计的txt下载到windows下，将class_num_label_final进行可视化
    class_num_label_final.sort()
    plt.plot(class_num_label_final)
    plt.show()

if __name__ == '__main__':
    print('正在读取h5文件......')
    class_label_final = []
    class_num_label_final = []
    data_num_final = 0
    data_class_final = 0
    max_class_num_final = 0
    min_class_num_final = 10000
    mean_class_num_final = 0.0
    mean_len_final = 0.0
    for i in range(6):
        f = h5py.File('/home/wangbc1/OCR/data/data_final_%d.h5' % (i+1), 'w')
        label = np.vstack((f['label_train'][:], f['label_val'][:]))
        class_label, class_num_label, data_num, data_class, max_class_num, min_class_num, mean_class_num, mean_len = analyselabel(
            label)
        for j in range(len(class_label)):
            if class_label[j] in class_label_final:
                class_num_label_final[class_label_final.index(class_label[j])] += class_num_label[j]
            else:
                class_label_final.append(class_label[j])
                class_num_label_final.append(class_num_label[j])
        data_num_final += data_num
        max_class_num_final = max(max_class_num_final,max_class_num)
        min_class_num_final = min(min_class_num_final,min_class_num)
        mean_class_num_final += mean_class_num
        mean_len_final += mean_len

    data_class_final = len(class_label_final)
    mean_class_num_final /= 6
    mean_len_final /= 6

    with open('/home/wangbc1/OCR/Analyse_data.txt','w') as f:
        f.write("  data_num_final  : %d\n" % data_num_final)
        f.write("  max_class_num_final  : %d\n" % max_class_num_final)
        f.write("  min_class_num_final  : %d\n" % min_class_num_final)
        f.write("  mean_class_num_final  : %f\n" % mean_class_num_final)
        f.write("  mean_len_final  : %f\n" % mean_len_final)
        f.write("  class_label_final  : %s\n" % str(class_label_final))
        f.write("  class_num_label_final  : %s\n" % str(class_num_label_final))

    # drawplot(class_num_label_final)


