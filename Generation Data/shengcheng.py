# -*- coding: utf-8 -*-
# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import h5py
from common import *
import cv2

def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print "total number of images : ", colorize(Color.RED, len(dsets), highlight=True)
    num = 1
    with open('/home/wangbc1/OCR/SynthText_Chinese_version-master/SynthText_Chinese_version-master/english/image3/label.txt','w') as f:
        for k in dsets:
            sign = 0
            rgb = db['data'][k][...]
            #charBB = db['data'][k].attrs['charBB']
            wordBB = db['data'][k].attrs['wordBB']
            txt = db['data'][k].attrs['txt']
            for i in range(len(txt)):
                newtxt = txt[i].split('\n')
                for j in range(len(newtxt)):
                    bb = wordBB[:, :, sign]
                    sign+=1
                    if num % 1000 == 0:
                        print num
                        print min(bb[1,:]),max(bb[1,:]),min(bb[0,:]),max(bb[0,:])
                    if min(bb[1,:])<0 or max(bb[1,:])<0 or min(bb[0,:])<0 or max(bb[0,:])<0:
                        continue
                    else:
                        temp = rgb[min(bb[1,:]):max(bb[1,:]),min(bb[0,:]):max(bb[0,:])]
                        cv2.imwrite('/home/wangbc1/OCR/SynthText_Chinese_version-master/SynthText_Chinese_version-master/english/image3/%d.jpg' % num, temp)
                        f.write('%d.jpg ' % num + newtxt[j].encode('utf-8') + ' \n')
                        num += 1
        db.close()

if __name__=='__main__':
    main('/home/wangbc1/OCR/SynthText_Chinese_version-master/SynthText_Chinese_version-master/english/SynthText_8000_3.h5')

