# -*- coding: utf-8 -*-
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
    img = cv2.imread(r'C:\Users\wangbc1\Desktop\OCR\1(4).jpg' )
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
    g1 = tf.Graph()
    with g1.as_default():
        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(r'C:\Users\wangbc1\Desktop\OCR\model\detection\22\CNN-Model-12 (2).meta')
        saver.restore(sess, r'C:\Users\wangbc1\Desktop\OCR\model\detection\22\CNN-Model-12 (2)')
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
            detection_result = sess.run(train_result, feed_dict={x: temp_pic})
            if detection_result[0] == 0:
                regions_list.pop(base_num)
            else:
                base_num += 1
        sess.close()
    # if len(regions_list)==0:
    #     return
    # 除去包含关系的区域，保留大区域
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

    # 识别。注意：label必须随着数据实时更新，不然无法解出结果。
    g2 = tf.Graph()
    with g2.as_default():
        sess2 = tf.InteractiveSession()
        saver2 = tf.train.import_meta_graph(r'C:\Users\wangbc1\Desktop\OCR\model\recognition\CNN+LSTM+CTC-Model-32.meta')
        saver2.restore(sess2, r'C:\Users\wangbc1\Desktop\OCR\model\recognition\CNN+LSTM+CTC-Model-32')
        decoded_final = tf.get_collection('pred_network')
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('x').outputs[0]
        is_train = graph.get_operation_by_name('is_train').outputs[0]
        seq_length = graph.get_operation_by_name('seq_length').outputs[0]
        label = '诚娇廖更纳加奉公一计与路房原其全安久钟不影关地问炎韵月田节大心东愉汇科业里航晏字平录先彤有岳绍在文定名水理让发为含相七日蒂栾栗州滑了司宰兴保用如向国开民何长施羽中人世古多倪庚赛觉溢潮夕双单锦杜能效霜然晁放步新合也青天时和巍亨鉴体作化解腾师小程海引二裴修图胡许基柴研律再李骁章豆敏学会格查来焕舜狐实五宇美云九祺忻本钦进季于玖未嵩令风行支蓉钊松成可鹤从政球韫圆年米友区自兹流留同星湖三建儿震力情训濮商十召管六露党高骐央孝门男西项声景前富朋鱼波健方谈别炳招毛结任潘比扎选哈楷亿明婵凡达澹焰祁柏员禄怡龙白生闯起细聚上渊艾记蕙普朗光蔷万幻吉子闻井好校营占农经张谨堂玉信姬意宝尔钰艺特都荣倚登奇涵符道菊虹仲众南释北标茗贲妍卫英治元领穗飞火温果奕迪泳彩旭淑争兆柱传至包内临红功衡老布载苹娄库欧边瞳得继士策言干音举黎寒蒋观祖位主京帝才黄炯四园牧容汉济应花沣以茅詹译宁郑余鄂珑群琨环常周女简范法遥汝奥直贞钧旺依之续展品清统涌滨冈咏迅阔列川利易毓空术苟广鲍康想尤知裕帮雨娟苑后蔡帕赫升梁强由劭费甘通深承烨神华育静运庆曼克代官晟霁又菡乜郁菜现乐维左吾硕积联玛则见振班赵印籍财莎勋励将帅纪岭滕冠隆介姗锴琰繁寅若毋思轲重筱委晋志徽呈口村劳池正家复岛镇冰频采八缪熹楠穆傲台恒葛永烟桦书瀚超铮望卡革索戚岐山微陈驹德昌河宪玲源语照顺晓尚社孟浪军笑鲁绿颖乌孔巴早炀昝铁瑗权韦靓武宫坡培佩迟翼姣金崇兵根欢恬崔扶彭浦昀弋娣致初苓城逢灵密陶琛沛岱诺征冷素蓟涓挺史焱尘苏樊邓义蒲豪太紫珉墨渭慕璇嵘刘潞洛须兮林善煊逯宸远峥娥谷斐洋非威候朝瑞头仪车浩茂戴邹翁无喜淞砚弈翟暨舟珲棠媛尉韩危立阳少申居游歌舞沙劲阿郭霏忠材灿今枝奎於翌侯秀存岗璨煌键益连睿均胥煜丽莲战显宾巧博凯斌彬樱福净阴雷楚罡菁爱融珂宗毅必杨莘菱淼茹颂戈仰封椿寿铃嫣嵇江石银靳芷兰智臧霍庭纯种绘团香蓝汲辰齐念耶旗闵婉顾玟董俭驰昕儒丝彰吟骄守燕满邱泽赞晔创翔粉楼土习桢桂谦延坚蔚璩殿王麻丘柯骆添予谌亮莺真轶翊冼妮荆陵孜千靖杰栋灏煦芬彪郡良隗雯沐曹彦琪潼锋歆辕烈朴钱娃邵璞忆弟绮君佳霞逊珍那曲夏慎树丛臻述晨叶骋郝增侍嫦圣恺百卿狄廉终饶改乃訾拥佐颜榆暴贾罗托邴春耀拓蓓寇宣萌眉付曾颢澎纲刑皓典储淳妙毕盖鸿娴贤碧霄映牛恩荔阙唐舒瞿汐黑芮鸣宏严瑛庾夫纬侠竞姜逸锁惠攀尼阮鹰亚勒刚丰养冶辉麦港卜段薪历喻猛仁衍璐贝宽魏伍琳姿却仙森雅朵龚礼汶筠玮鞠姆宦麟莉仕仓秋尧廷隋梦祥堵荟炫栩甜沂鹿伯徐匡欣默衣禅葳馨吴马帆咸钥庄凝泉谊陆弗乔席鸽胤璋卢逄萧聂冲滔汪祯鑫啸巢瀛木厚媚郜惜珠唯骊滢勤蕾镜颐佑桃贻昭蔓甲蕴蓬铎旖糜迈粟羊桥昱翎晖峻屏莹滟溥臣约盛峰嘉胜贺傅钢佟锐苗俞鲜寻怀巨薄愿亓耘珈赐莱珏润玄誉嗣爽赢漪涂烽赖苇澜梓肇屠鹭植竺邦冯耕沁巩悠湘洪锟谋迎奈邝稳雄曦禹湛诸燃贵屹壮豹邬童芳懿泰茜枫朔梅丞慈顿琬沈萱野冀隽俐芯熠禾熊蒙闫琦婧莫姝涛母启昊泓越塔扬杉弓霆岩芝勃甄佘伦皇庞斯谭伟宜辛溪晰奚献萨洲允柔氏扈郗钮敦汀尊仝汤宓铄晶牟聿丙朱淦妤戎婕瑾阎昶侃塞萍豫韬晴亭妃轩厉垒岑沅瑶皮伶晗优甫端曙拉捷凌渝骥玫笛阚鸾希妹尹焦刁舰乙茵函伊娜翠薇骏俊羿洁宿颍葵琴耿蔺炜秦幼绪娅瓦雪伏雍卓屈祝勾薛芙熙慧惟咪郦跃吕肖泊津殷漫俏孙沃凤邢鄢桐婷诗芸馥恭剑龄桑弘桓翰敖郎魁玺蕊巫旦鹏贡芹岚铖瑜畅榕练菲昆麒愚娉藤柳哲烁坦潇珩乾锡珊芊淋谢澄澍操勇悦虎荀宛倩窦昂袁琼雁杏卉杭竣虞丹敬叔忱裘梵荷卞铭盼淇俪渡韶霖槐艳蝶旋骞盈丁磊邰纤冬暄丕骅鼎姚芦坤浚秉埃褚宋竹佰崎缘梨亦束阁棋聪冉苍幸仇覃殳玥仵蒯陳钭楊旸喆珺厍赟旻芃黃劉酆翀祎垚張珮彧吳璟堃姍'
        recognition_list = []
        for i in range(len(regions_list)):
            temp_pic = img[regions_list[i][1]:regions_list[i][3]+1,regions_list[i][0]:regions_list[i][2]+1,:]
            temp_pic = cv2.resize(temp_pic, (100, 32))
            cv2.imshow('rect', temp_pic)
            cv2.waitKey(0)
            temp_pic = temp_pic / 255.0
            temp_pic = temp_pic[np.newaxis, :, :, :]
            recognition_result = sess2.run(decoded_final, feed_dict={x: temp_pic, is_train: np.array([1]), seq_length: np.array([24])})
            final = ''
            for i in recognition_result[0][0]:
                final += label[i]
            recognition_list.append(final)
        sess2.close()
    print(recognition_list)
    # 画图
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
