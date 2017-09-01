import numpy as np
import cv2
import time
from skimage import dtype_limits,measure,color
import tensorflow as tf


# 计算图像的对比度
def contrast(image):
    dlimits = dtype_limits(image, clip_negative=False)
    limits = np.percentile(image, [1,99])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
    return ratio

if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\wangbc1\Desktop\OCR\1(5).jpg' )
    # start = time.clock()

    # 规范图像的尺寸
    threshold = 420
    if min(img.shape[0:2]) > threshold:
        if img.shape.index(min(img.shape[0:2])) == 0:
            img = cv2.resize(img, (threshold, int(img.shape[1] / img.shape[0] * threshold)))
        else:
            img = cv2.resize(img, (int(img.shape[0] / img.shape[1] * threshold), threshold))
    elif min(img.shape[0:2]) < threshold:
        if img.shape.index(min(img.shape[0:2])) == 0:
            img = cv2.resize(img, (threshold, int(img.shape[1] / img.shape[0] * threshold)))
        else:
            img = cv2.resize(img, (int(img.shape[0] / img.shape[1] * threshold), threshold))

    # 先计算灰度图的对比度，如果小于0.67，就将(R,G,B,gray)中对比度最大的用在MSER算法中，并且区域选择的标准会很松。
    # 如果灰度图对比度大于等于0.67，就用灰度图做MSER
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast_list = []
    contrast_list.append(contrast(gray))

    if contrast_list[0] >= 0.67:
        mser = cv2.MSER_create(_min_area=60, _max_area=int(img.shape[0] * img.shape[1] / 25))
        regions = mser.detectRegions(gray, None)
    else:
        mser = cv2.MSER_create(_delta=1, _max_variation=0.5, _min_area=60,
                               _max_area=int(img.shape[0] * img.shape[1] / 25))
        contrast_list.append(contrast(img[:, :, 0]))
        contrast_list.append(contrast(img[:, :, 1]))
        contrast_list.append(contrast(img[:, :, 2]))
        if contrast_list.index(max(contrast_list)) == 0:
            regions = mser.detectRegions(gray, None)
        else:
            regions = mser.detectRegions(img[:, :, contrast_list.index(max(contrast_list)) - 1], None)

    # MSER有很多重叠的区域，下面对重叠的区域进行去重
    regions_list = []
    base_num = 0
    while base_num <= len(regions) - 1:
        min_cols_one = np.min(regions[base_num][:, 0])
        max_cols_one = np.max(regions[base_num][:, 0])
        min_rows_one = np.min(regions[base_num][:, 1])
        max_rows_one = np.max(regions[base_num][:, 1])
        regions_list.append([min_cols_one, min_rows_one, max_cols_one, max_rows_one])
        base_num += 1

    base_num = 0
    while base_num + 1 <= len(regions_list) - 1:
        min_cols_one = regions_list[base_num][0]
        max_cols_one = regions_list[base_num][2]
        min_rows_one = regions_list[base_num][1]
        max_rows_one = regions_list[base_num][3]

        min_cols_two = regions_list[base_num + 1][0]
        max_cols_two = regions_list[base_num + 1][2]
        min_rows_two = regions_list[base_num + 1][1]
        max_rows_two = regions_list[base_num + 1][3]

        minx = max(min_rows_one, min_rows_two)
        miny = max(min_cols_one, min_cols_two)
        maxx = min(max_rows_one, max_rows_two)
        maxy = min(max_cols_one, max_cols_two)

        if minx > maxx or miny > maxy:
            oe = 0

        else:
            oe = ((maxx - minx) * (maxy - miny)) / \
                 ((max_cols_one - min_cols_one) * (max_rows_one - min_rows_one) + (max_cols_two - min_cols_two) * (
                 max_rows_two - min_rows_two) - (maxx - minx) * (maxy - miny))

        if oe >= 0.85:
            if (max_cols_one - min_cols_one) * (max_rows_one - min_rows_one) >= (max_cols_two - min_cols_two) * (
                max_rows_two - min_rows_two):
                regions_list.pop(base_num + 1)
            else:
                regions_list.pop(base_num)
        else:
            base_num = base_num + 1

    base_num = 0
    jump_num = 1
    while base_num + 1 <= len(regions_list) - 1:
        while base_num + jump_num <= len(regions_list) - 1:
            min_cols_one = regions_list[base_num][0]
            max_cols_one = regions_list[base_num][2]
            min_rows_one = regions_list[base_num][1]
            max_rows_one = regions_list[base_num][3]

            min_cols_two = regions_list[base_num + jump_num][0]
            max_cols_two = regions_list[base_num + jump_num][2]
            min_rows_two = regions_list[base_num + jump_num][1]
            max_rows_two = regions_list[base_num + jump_num][3]

            minx = max(min_rows_one, min_rows_two)
            miny = max(min_cols_one, min_cols_two)
            maxx = min(max_rows_one, max_rows_two)
            maxy = min(max_cols_one, max_cols_two)

            if minx <= maxx and miny <= maxy:
                oe = ((maxx - minx) * (maxy - miny)) / \
                     ((max_cols_one - min_cols_one) * (max_rows_one - min_rows_one) + (
                     max_cols_two - min_cols_two) * (max_rows_two - min_rows_two) - (maxx - minx) * (maxy - miny))
                if oe >= 0.65:
                    regions_list[base_num].clear()
                    regions_list[base_num].extend(
                        [int((min_cols_one + min_cols_two) / 2), int((min_rows_one + min_rows_two) / 2),
                         int((max_cols_one + max_cols_two) / 2), int((max_rows_one + max_rows_two) / 2)])
                    regions_list.pop(base_num + jump_num)
                else:
                    jump_num += 1
            else:
                jump_num += 1
        base_num += 1
        jump_num = 1

    # 把剩下的区域转化为28*28，之后传入CNN进行预测，识别文字区域与非文字区域
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(r'C:\Users\wangbc1\Desktop\OCR\model\22\CNN-Model-12 (2).meta')
    saver.restore(sess, r'C:\Users\wangbc1\Desktop\OCR\model\22\CNN-Model-12 (2)')
    train_result = tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x').outputs[0]

    base_num = 0
    while base_num <= len(regions_list) - 1:
        min_cols = regions_list[base_num][0]
        max_cols = regions_list[base_num][2]
        min_rows = regions_list[base_num][1]
        max_rows = regions_list[base_num][3]
        temp_pic = img[min_rows:max_rows + 1, min_cols:max_cols + 1, :]
        temp_pic = cv2.resize(temp_pic, (28, 28))
        temp_pic = temp_pic/255.0
        temp_pic = temp_pic[np.newaxis, :, :, :]
        result = sess.run(train_result, feed_dict={x: temp_pic})
        if result[0] == 0:
            regions_list.pop(base_num)
        else:
            base_num += 1

    #出去包含关系的区域，保留大区域
    base_num = 0
    jump_num = 1
    while base_num + 1 <= len(regions_list) - 1:
        while base_num + jump_num <= len(regions_list) - 1:
            min_cols_one = regions_list[base_num][0]
            max_cols_one = regions_list[base_num][2]
            min_rows_one = regions_list[base_num][1]
            max_rows_one = regions_list[base_num][3]

            min_cols_two = regions_list[base_num + jump_num][0]
            max_cols_two = regions_list[base_num + jump_num][2]
            min_rows_two = regions_list[base_num + jump_num][1]
            max_rows_two = regions_list[base_num + jump_num][3]

            if min_rows_one <= min_rows_two and min_cols_one <= min_cols_two and max_rows_one >= max_rows_two and max_cols_one >= max_cols_two:
                regions_list.pop(base_num + jump_num)
                continue

            elif min_rows_one >= min_rows_two and min_cols_one >= min_cols_two and max_rows_one <= max_rows_two and max_cols_one <= max_cols_two:
                regions_list.pop(base_num)
                jump_num = 1
                continue
            else:
                jump_num += 1
        base_num += 1
        jump_num = 1

    #行归并
    base_num = 0
    regions_list = np.array(regions_list)
    arr = regions_list.argsort(0)
    regions_list = regions_list[arr[:,1]]
    regions_list = list(regions_list)

    while base_num+1<=len(regions_list)-1:
        if  abs(regions_list[base_num][1]-regions_list[base_num+1][1])<=5 and abs(regions_list[base_num][3]-regions_list[base_num+1][3])<=5:
            regions_list[base_num] = [min(regions_list[base_num][0],regions_list[base_num+1][0]),min(regions_list[base_num][1],regions_list[base_num+1][1]),max(regions_list[base_num][2],regions_list[base_num+1][2]),max(regions_list[base_num][3],regions_list[base_num+1][3])]
            regions_list.pop(base_num+1)
        else:
            base_num += 1

    #文字分割
    # base_num = 0
    # final_region_list= []
    # temp_list = []
    # while base_num<=len(regions_list)-1:
    #     temp_pic = img[regions_list[base_num][1]:regions_list[base_num][3] + 1, regions_list[base_num][0]:regions_list[base_num][2] + 1, :]
    #     gray = cv2.cvtColor(temp_pic, cv2.COLOR_BGR2GRAY)
    #     ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #     image = measure.label(thresh,connectivity=1)
    #     region_props = measure.regionprops(image)
    #     for i in range(len(region_props)):
    #         temp_list.append([region_props[i].bbox[1]+regions_list[base_num][0],region_props[i].bbox[0]+regions_list[base_num][1],
    #                           region_props[i].bbox[3] + regions_list[base_num][0],region_props[i].bbox[2] + regions_list[base_num][1]])
    #     final_region_list.append(temp_list.copy())
    #     temp_list.clear()
    #     base_num+=1

    # i = 0
    # j = 0
    # while i <= len(final_region_list)-1:
    #     while j+1 <= len(final_region_list[i])-1:
    #         min_cols_one = final_region_list[i][j][0]
    #         max_cols_one = final_region_list[i][j][2]
    #         min_rows_one = final_region_list[i][j][1]
    #         max_rows_one = final_region_list[i][j][3]
    #
    #         min_cols_two = final_region_list[i][j+1][0]
    #         max_cols_two = final_region_list[i][j+1][2]
    #         min_rows_two = final_region_list[i][j+1][1]
    #         max_rows_two = final_region_list[i][j+1][3]
    #
    #         minx = max(min_rows_one, min_rows_two)
    #         miny = max(min_cols_one, min_cols_two)
    #         maxx = min(max_rows_one, max_rows_two)
    #         maxy = min(max_cols_one, max_cols_two)
    #
    #         if minx <= maxx and miny <= maxy:
    #             final_region_list[i][j] = [min(min_cols_one,min_cols_two),min(min_rows_one,min_rows_two),max(max_cols_one,max_cols_two),
    #                                        max(max_rows_one,max_rows_two)]
    #             final_region_list[i].pop(j+1)
    #         else:
    #             j+=1
    #     i+=1


    # i = 0
    # j = 0
    # while i <= len(final_region_list)-1:
    #     while j <= len(final_region_list[i])-1:
    #         min_cols = final_region_list[i][j][0]
    #         max_cols = final_region_list[i][j][2]
    #         min_rows = final_region_list[i][j][1]
    #         max_rows = final_region_list[i][j][3]
    #         temp_pic = img[min_rows:max_rows + 1, min_cols:max_cols + 1, :]
    #         temp_pic = cv2.resize(temp_pic, (28, 28))
    #         temp_pic = temp_pic / 255.0
    #         temp_pic = temp_pic[np.newaxis, :, :, :]
    #         result = sess.run(train_result, feed_dict={x: temp_pic})
    #         if result[0] == 0:
    #             final_region_list[i].pop(j)
    #         else:
    #             j+=1
    #     i+=1

    # for i in range(len(final_region_list)):
    #     for j in range(len(final_region_list[i])):
    #         min_cols = final_region_list[i][j][0]
    #         max_cols = final_region_list[i][j][2]
    #         min_rows = final_region_list[i][j][1]
    #         max_rows = final_region_list[i][j][3]
    #         cv2.rectangle(img,(min_cols,min_rows),(max_cols,max_rows),(0,255,0) )

    base_num = 0
    while base_num<=len(regions_list)-1:
        min_cols = regions_list[base_num][0]
        max_cols = regions_list[base_num][2]
        min_rows = regions_list[base_num][1]
        max_rows = regions_list[base_num][3]
        cv2.rectangle(img,(min_cols,min_rows),(max_cols,max_rows),(0,255,0) )
        base_num+=1

    cv2.namedWindow("OCR",0)
    cv2.imshow('OCR', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sess.close()