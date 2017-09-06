#!/usr/bin/python2.7.13
# -*- coding: utf-8 -*-
"""
Created on  2017/7/10 16:05

@author: zhoukang
"""
import cv2
import os
import numpy as np

def calcAHashCode(image):
    resized = cv2.resize(image, (8,8), interpolation=cv2.INTER_LINEAR)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mean = np.mean(resized)
    code = np.zeros((8,8), dtype=np.int32)
    code[resized >= mean] = 1
    return code.reshape((1,64))

def calcPHashCode(image):
    resized = cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resized = cv2.dct(resized)
    resized = resized[:8,:8]
    mean = np.mean(resized[:8,:8])
    code = np.zeros((8,8), dtype=np.int32)
    code[resized >= mean] = 1
    return code.reshape((1,64))

def isDuplication(code, code_book):
    for i in range(code_book.shape[0]):
        diff = np.sum(np.logical_xor(code, code_book[i,:]))
        if diff <= 8:
            return True,i
    return False,-1

if __name__ == '__main__':
    path = r'E:\OCRImage\trainImage\6'
    files = os.listdir(path)
    code_book = np.zeros((len(files), 64))
    #code_book = None
    names = []
    delete_list = []
    cnt=0
    for i, f in enumerate(files):
        if (i+1) % 1000 == 0:
            print('processed %d images.'%(i+1))
        name = os.path.join(path, f)
        if f.split('.')[-1].lower() in ['jpg', 'png', 'jpeg', 'bmp']:
            image = cv2.imread(name, 1)
            code = calcAHashCode(image)
            # code = calcPHashCode(image)
            if cnt > 0:
                duplication,index = isDuplication(code, code_book[0:cnt,:])
                if not duplication:
                    #### todo:too slow! pre-malloc memory space
                    #code_book = np.concatenate([code_book, code], axis=0)
                    code_book[cnt,:] = code
                    cnt += 1
                    names.append(f)
                else:
                    delete_list.append(f)
                    if False:
                        cv2.namedWindow('code', 0)
                        cv2.imshow('code', image)
                        image = cv2.imread(os.path.join(path, names[index]), 1)
                        cv2.namedWindow('code_book', 0)
                        cv2.imshow('code_book', image)
                        cv2.waitKey(0)
            else:
                names.append(f)
                code_book[0,:] = code
                cnt += 1
    np.save('delete.npy', np.asarray(delete_list))
    for f in delete_list:
        os.remove(os.path.join(path, f))
    print('Done.')