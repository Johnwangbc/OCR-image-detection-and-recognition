import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
import wget, tarfile
import cv2
from PIL import Image




def add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path):
  db=h5py.File(DB_FNAME,'w')
  depth_db=h5py.File(more_depth_path,'r')
  seg_db=h5py.File(more_seg_path,'r')
  db.create_group('image')
  db.create_group('depth')
  db.create_group('seg')

  num = 1
  f = open(r"E:\OCRImage\imnames.cp", "r")
  lines = f.readlines()

  for imname in lines:#os.listdir(more_img_file_path):
    imname = imname.strip('\n')
    imname = imname[2:]
    if imname.endswith('.jpg'):
      full_path=more_img_file_path+imname
      print full_path,imname,num
      
      j=Image.open(full_path)
      imgSize=j.size
      rawData=j.tostring()
      try:
        img=Image.fromstring('RGB',imgSize,rawData)
      except:
        continue
      #img = img.astype('uint16')
      try:
        db['depth'].create_dataset(imname, data=depth_db[imname])
        db['image'].create_dataset(imname,data=img)
        db['seg'].create_dataset(imname,data=seg_db['mask'][imname])
        db['seg'][imname].attrs['area']=seg_db['mask'][imname].attrs['area']
        db['seg'][imname].attrs['label']=seg_db['mask'][imname].attrs['label']
        num+=1
      except:
        continue
  db.close()
  depth_db.close()
  seg_db.close()
  f.close()


# path to the data-file, containing image, depth and segmentation:
DB_FNAME = r'D:\dset_8000.h5'

#add more data into the dset
more_depth_path=r'E:\OCRImage\depth.h5'
more_seg_path=r'E:\OCRImage\seg.h5'
more_img_file_path='E:\\OCRImage\\bg_img\\bg_img\\'

add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path)
