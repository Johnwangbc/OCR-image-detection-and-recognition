# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

def deletepartdata(class_label,class_num_label,data,label,seqlength):
    # 找出标签中出现次数小于阈值的文字，把包含这些文字的数据从data,label,seqlength中去除
    delete_list = []
    print('处理delete_list')
    for i in range(len(class_num_label)):
        if class_num_label[i] < 15:
            delete_list.append(class_label[i])

    data = list(data)
    label = list(label)
    seqlength = list(seqlength)
    i = 0
    j = 0
    while i <= len(label)-1:
        print (i)
        while j <= len(label[i])-1:
            if label[i][j] in delete_list:
                j = 0
                data.pop(i)
                label.pop(i)
                seqlength.pop(i)
                break
            else:
                j += 1
        i += 1
        j = 0
    return np.asarray(data,np.float32),np.asarray(label,np.int32),np.asarray(seqlength,np.int32)

if __name__ == '__main__':
    print('loading......')
    #label = np.load(r'C:\Users\wangbc1\Desktop\OCR\data\label3.npy')
    data = np.load('/home/wangbc1/OCR/data/data3.npy')
    label = np.load('/home/wangbc1/OCR/data/label3.npy')
    seqlength = np.load('/home/wangbc1/OCR/data/seqlength3.npy')
    data_num = 0
    data_class = 0
    max_class_num = 0
    min_class_num = 0
    mean_class_num = 0.0
    mean_len = 0.0
    class_label = []
    class_num_label = []

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
    # class_num_label.sort()
    # plt.plot(class_num_label)
    # plt.show()
    # c = 1
    data4, label4, seqlength4 = deletepartdata(class_label,class_num_label,data,label,seqlength)
    np.save('/home/wangbc1/OCR/data/data4.npy',data)
    np.save('/home/wangbc1/OCR/data/label4.npy', label)
    np.save('/home/wangbc1/OCR/data/seqlength4.npy', seqlength)

    # 以下是统计样本一行最多多少字
    # max_len = 0
    # with open(r'C:\Users\wangbc1\Desktop\OCR\data\label (5).txt', 'r',encoding='utf-8') as f:
    #     a = f.readlines()
    #     for i in range(len(a)):
    #         b = a[i].split()[1]
    #         max_len = max(max_len,len(b))
    #     print(max_len)

