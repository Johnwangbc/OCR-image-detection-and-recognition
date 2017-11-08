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
    img = cv2.imread(r'C:\Users\wangbc1\Desktop\OCR\1(2).jpg' )
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
        detection_result = sess.run(train_result, feed_dict={x: temp_pic})
        if detection_result[0] == 0:
            regions_list.pop(base_num)
        else:
            base_num += 1

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

    # 识别
    saver = tf.train.import_meta_graph(r'C:\Users\wangbc1\Desktop\OCR\model\recognition\CNN+LSTM+CTC-Model-2.meta')
    saver.restore(sess, r'C:\Users\wangbc1\Desktop\OCR\model\recognition\CNN+LSTM+CTC-Model-2')
    decoded_final = tf.get_collection('pred_network')
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x').outputs[0]
    is_train = graph.get_operation_by_name('is_train').outputs[0]
    seq_length = graph.get_operation_by_name('seq_length').outputs[0]
    label = '绚诚娇溜题者廖更纳加奉公一就汴计与路房原妇其骑刈全消昏傈安久钟不影处蜿资关椤地专问忖票炎韵要月田节鄙捌备拳伺眼网盎大傍心东愉汇科每业里航晏字平录先彤产督腴有象岳注绍在泺文定核名水过理让偷率等这发为含肥酉相七锛日蒂掰倒栾栗综涩州雌滑馀了机司宰兴矽抚保用沧秩如收息页疑埠姥异橹钇向下的椴沫国绥报开民何分凇长讥藏掏施羽中讲派嘟人提浼间世而古多倪唇饯庚首赛蜓味断制觉技替艰溢潮夕钺外摘枋动双单户枇确锦曜杜或能效霜然侗电晁放步鹃新杖蜂吒濂瞬评总隍对独合也是府青天诲墙组滴级邀帘示已时仄和遨店疫持巍境只亨目鉴崤闲体泄杂作般轰化解迂蛭璀腾告版服省师小规程线海办引二桧牌砺洄裴修图胡许事基柴呼食研奶律蛋因葆察戏褒戒再李骁工貂油鹅章啄休场给睡纷豆器说敏学会浒设格廓查来霓室溆诡寥焕舜柒狐回戟砾厄实翩尿五入径惭股宇美期云九祺靠系企阊暂蚕忻豁本羹执条钦H獒限进季楦于芘玖铋茯未答粘括样精欠矢甥帷嵩扣令仔风皈行支部蓉站救钊汗松嫌成可鹤院从交政活调球局验髌第韫串到圆年米友检区看自敢刃个兹弄流留同没齿星聆轼湖什三建儿椋汕震颧鲤跟力情璺铨陪务指族训滦濮扒商箱十召慷辗所莞管护臭横嗓接侦六露党驾剖高侬妪幂猗骐央孝筝缰门男西项句谙秃教声呐景前富嘴鳌稀免朋啬睐去赈鱼住肩愕速波厅健茼厥鲟谅投攸炔数方击谈绩别愫僚躬鹧炳招喇膨泵蹦毛结谱识陕粽婚拟构且搜任潘比郢妨醪陀桔碘扎选哈楷亿明缆脯监睫逻婵共赴凡惦及达揖谩澹焰番祁柏员禄怡峤龙白生闯起细装谕竟聚钙上导渊按艾挡耒饪臀记邮蕙受各医普滇朗茸带翻光堤墟蔷万幻昧盏亘吉铰请子假闻税井诩哨嫂好面琐校鬣营访占农缀否经钚棵张亟吏茶谨论堂玉信吧瞠乡姬寺咬苄皿意赉宝尔钰艺特都荣倚登荐丧奇涵炭近符傩感道着菊虹仲众濯颞眺南释北标既茗整撼迤贲拒某妍卫英矶藩治他元领遮穗蛾飞荒棺劫么市火温棚洼转果奕卸迪伸泳斗邡侄涨屯萋胭枞惧冒彩斜手随旭淑妞形菌争驯歹兆柱传至包内响临红功弩衡寂禁老棍耆织害氵布载靥嗬苹咨娄库雉榜帜套瑚亲簸欧边腿旮吹瞳得镓梗厨继漾愣憨士策窑抑躯襟参贸言干绸鳄穷藜音折详举悍甸黎死罩迁寒驷袖媒蒋掘模纠恣观祖位稿主澧跌筏京锏帝贴证糠才黄鲸略炯饱四出园犀牧容汉杆浈瑷造虫瘩怪驴济应花沣谔夙旅价矿以考u晒巡茅准瓴詹仟译桌混宁郑抿余鄂饴攒珑群阖岔琨藓预环洮岌杲最常囡周踊女鼓袭简范遐疏粱禧法箔斤遥汝奥直贞撑置集逗钧恙躁唤旺待脾购吗依盲度瘿蠖之镗拇簧续款展表剔品钻损清锶统涌寸滨贪链冈伎迥咏吁览防迅失汾阔逵绀蔑列川凭努熨揪利俱绉抢我即责膦易毓鹊刹玷岿空嘞排术估锷违们苟铜播肘件烫审鲂广像铌铟巳鲍康憧色恢想尤知SYFDA峄裕帮握难墒沮雨叁缥藐湫娟苑稠颛簇后阕闭蕤怎佞嘤蔡舱螯帕赫昵升烬岫疵蜻蕨隶烛械丑盂梁强鲛由揉劭龟钩孛费妻漂求阑崖甘通深补坎床承吼量暇钼烨阂擎脱称P神属矗华届葑育蛰佼静槎运鳗庆曼克代官此蚌晟例础榛副测缢迹霁身岁赭又菡乜雾板读陷徉贯郁虑变钓菜圾现琢式乐维渔浜左吾脑钡警拴偌漱湿硕止魄积燥联玛则窿见振畿送班您赵印讨籍谡舌崧汽蔽沪酥绒怖财帖肱莎勋羔霸励将帅渠纪婴娩岭厘滕吻伤坝冠戊隆介涧物黍并姗奢蹑锴命病琰眭迩艘繁寅若毋思诉类诈燮轲狂重反职筱县委绣奖晋濉志徽肠呈坻口片碰几村柿劳料获惕晕厌号罢池正鏖煨家棕复尝懋锅岛扰队坠瘾钬卧镇冰彷频黯据垄采八缪型熹砰楠襁箐但嘶绳啤拍穆傲洗塘怔筛台恒喂葛永烟酒桦书砂缉态瀚袄圳轻超榧姒奘铮右望卡附做革索戚唁岐偎坛莨山殊微陈爨推驹呤卤嘻侵郓酌德摇被慨殡羸昌泡戛鞋河宪沿玲翅源铅语照邯址荃佬顺鸳町霭瓢夸椁晓酿侏噎湍签离午尚社锤背孟使浪缦潍鞅军姹笑鳟鲁孽钜绿洱礴焯椰颖囔乌孔巴互性椽聘昨早暮胶炀隧彗昝铁呓氽藉瑗姨权胱韦蜜酋楝砝毁靓歙锲究屋喳骨辨碑武鸠宫辜烊适坡培佩供走蜈迟翼况姣凛浔吃飘债犟金促苛崇坂莳畔绂兵斋根亢欢恬崔餐快扶濒缠当彭驭浦篮昀锆秸钳弋娣夷拱致障隐初娓抉汩累助苓昙押破城郧逢瞻溱婿璧萃姻灵炉密陶谬衔点琛沛枳层岱诺埂征冷裁打素逞聊激腱萘飒蓟吆取涓矩曝挺揣座你史舵焱尘苏笈脚溉榨诵樊邓焊义庶儋蒲赦呷杞诠豪还试颓茉太除紫逃痴草充鳕珉祗墨渭烩蘸慕璇镶穴嵘恶险幕戳刘潞秣纾潜銮洛须罘销汞兮林厕质探划狸殚善煊烹锈逯宸泱柚远蹋嶙峥娥雀徵认镱谷贩勉鄯斐洋非祚泾诒威搭芍锥笺蓦候琊档礁沼荠朝凹瑞头仪畏突衲车浩气茂悖厢枕酝戴湾邹飚攘锂写宵翁岷无喜丈挑绛议槽具醇淞笃郴阅饼底壕砚弈询缕庹翟零筷暨舟闺甯撞茌蔼很珲棠角媛娲诽尉爵睬韩诰危镯立浏阳少盆擘匪申铣旯抖赘瓯居哮游锭茏歌坏甚秒舞沙仗劲潺阿燧郭霏忠材奂耐砀输岖媳氟极灿今枝奎药熄吨话额协喀壳埭视著於愧翌峁佛腹聋侯咎叟秀颇存罪哄岗扫栏钾羌己璨枭煌涸衿键镝益岢奏连夯睿冥均糖狞蹊稻爸胥煜丽璃跚灾垂樾濑乎莲犹撮战软络显鸢胸宾妲恕埔蝌份遇巧粒恰剥桡博讯凯堇阶滤卖斌骚彬兑樱舷两娱福仃差找桁净把阴污戬雷碓蕲楚罡焖妫咒仑闱尽邑菁爱贷沥鞑牡崴骤嗦订拮滓锻次坪杩箬融珂鹗宗枚降鸬妯堰盐毅必杨崃俺甬状莘货耸菱铸唏孚澳懒溅翘疙杷淼缙喊悉砻坷艇赁界谤纣宴晃茹归饭梢街抄鬟颂戈炒负仰客琉铢封卑珥椿镧窨鬲寿御袤铃萎砖餮裳孕嫣馗嵇恳江石褶冢阻羞银靳透敷芷它兰逑往坊甩沦忘菅剧智臧霍墅攻眯拢骠庭岙瓠缺泥迢郏沌纯秘种听绘固团香盗蓝拖旱荞铀血遏汲辰叩拽幅硬惶桀漠措泼齐念虚屁耶旗砦闵婉馆拭绅韧忏窝葺顾辞倜堆逆玟贱疾董倌锕淘莽俭笏择蟀粥驰逾案谪胫哩昕颚鲢绠躺鹄儒俨丝尕泌啊萸彰幺吟骄苣弦脊瑰诛镁析闪剪侧哟框守嬗燕狭缮概痧鲲俯笼扉挖满援邱扇歪便玑绦峡蛇叨泽胃斓喋怂猪该炕弥赞棣晔创铕镭稷弭翔粉履苘哦楼铂土锣瘟挣栉习享桢袅磨桂谦延坚蔚署谟猬钎恐嬉雒衅亏璩睹刻殿王算雕麻丘柯骆丸塍谚添鲈垓桎芥予镦谌窗醚菀亮搪莺蒿羁足真轶悬衷靛翊掩炅冼妮l谐稚荆擒犯陵虏浓崽刍陌傻孜千靖演矜钕煽杰渗伞栋俗泫戍罕沾灏煦芬叱榉湃蜀叉醒彪郡篷良垢隗弱陨峪砷掴颁胎雯绵沐隘篙暖曹陡栓填臼彦瓶琪潼哪鸡摩啦俟锋域耻疯纹毒绶痛忍爪赳歆辕烈册朴钱吮娃邵厮炽璞邃丐追词瓒忆轧芫谯喷弟半冕裙墉绮寝苔势顷褥切衮君佳嫒蚩霞佚洙逊镖暹顶碗獗铺废汨崩珍那杵曲纺夏薰傀闳淬姘舀拧卷恍讪篪赓乘灭盅沟慎挂饺杳树缨丛絮娌臻嗳篡述衰矛圈匕筹匿濞晨叶骋郝挚增侍描吖嫦蟒匾圣癞恺百曳需庖帏卿驿遗蹬鬓芎胳禽烦晌寄狄翡苒船廉终殇畦饶改拆萄乃訾桅溧拥纱铍骗蕃缬父佐栎醍蓄惆颜鲆榆猎敌暴谥贾罗玻缄芪落徒臾恿猩托邴肄牵春陛耀刊拓蓓邳寇枉淌啡湄兽酷萼碚濠萤夹旬梭琥昔勺晚孺宣摄冽旨萌忙蚤眉蟑付契瓜悼壁曾窕颢澎仿俑浑嵌浣乍碌乱玩葫箫纲围伐决伙漩瑟刑镳缓皓典畲檐塑洞倬储胴淳吐灼惺妙毕珐缈盖鸿磅谓娴苴唷蚣霹抨贤犬誓逍庠逼麓籼釉呜碧秧氩霄穸纨辟妈映完牛缴恩荔茆紊莓阙萁磐另辱鳐湮吩唐睦垠舒圜冗瞿芾匠僳汐菩漓黑霰浸濡窥毂蒡兢驻鹉芮雳厂臆猴鸣蚪栈箕羡渐莆捍眈哓蹼埕嚣骛宏淄斑噜严瑛椎诱压庾绞焘廿抡迄棘夫纬锹侠脐竞瀑孳骧遁姜颦荪滚萦伪逸粳锁矣役趣洒颔诏逐甭惠攀蹄泛尼拼阮鹰亚惑勒际肛爷刚丰养冶辉蔻画覆麦返醉皂擀酶凑粹悟硖港卜z杀涕舍铠弛段镐奠拂轴跛沉菇俎薪峦秭历盟菠寡液喻染抱赤猛谣仁尺烙衍架擦倏璐瑁币楞胖夔趸邛惴饕虔蝎哉贝宽炮扩饲籽魏菟锰伍末琳哚蛎呀姿鄞却歧仙森牒寤袒婆虢雅钉朵贼欲苞寰故龚坭咫礼兀睢汶铲烧绕诃浃钿哺柜璁腔洽簌筠镣玮鞠谁兼姆挥梯蝴谘漕躏宦弼b垌麟莉揭笙渎仕仓配怏抬错泯镊猿邪仍秋壹歇吵炼尧射柬廷胧凳隋肚浮梦祥株堵退鹫毽荟炫栩玳甜沂鹿顽伯爹赔徐匡欣狰缸雹蟆疤默沤啜痂衣禅i辽葳钗停沽棒馨颌肉吴硫悯劾娈马吊悌镑峭帆瀣涉咸滋泣翦拙癸钥尾庄凝泉婢渴谊乞陆鸦淮IBN晦弗乔庥葡尻席橡渣拿惩麋斛缃矮岘鸽姐催奔镒蠡摧钯胤柠拐璋鸥卢荡倾珀逄萧塾贮笆聂圃冲嵬滔笕值炙偶梆汪蔬鸯蹇敞绯仨祯谆梧鑫啸囹巢柄瀛筑沭暗苁脂蘖牢热木吸宠序泞拜檩厚毗媚朽担蝗橘畴祈盱隼郜惜珠铵焙琚唯咚骊丫滢勤棉呸咣淀隔蕾窈挨短匙镜赣墩馁颐抗酣佑搁哭递耷桃贻碣截瘦昭镌蔓甲蕴蓬散拾狼猷铎旖矾讳囊糜迈粟蚂紧鲳栽稼羊锄斟睁桥瓮祉醺鼻昱跳篱跷翎宅晖壑峻癫屏狠途憎祀莹滟佶溥臣约盛峰磁慵婪拦莅朕鹦粲裤琵堪谛嘉儡鳝郾驸妄胜贺傅钢栅庇恋匝邈尸锚粗佟蛟纵蚊郅绢锐苗俞篆淆鲜煎秽寻刺怀噶巨褰魅灶灌桉藕谜舸薄搀恽借牯渥愿亓耘杠钣珈幽赐稗晤莱泔肯菪腩疆骜腐倭珏粮亡润慰伽玄誉胆粼塬陇彼削嗣绾芽妗垭爽薏寨泠弹赢漪猫嘧涂恤圭茧烽痕巾赖凰腮畈偃苇澜艮换烘苕梓颉肇哗悄氤屠鹭植竺佯诣鲇邦移冯耕戌沁巩悠湘洪锟循谋腕钠焉迎伫急榷奈邝卯辄皲畹忧稳雄昼缩阈睑耗曦涅捏邕淖漉铝耦禹湛喽莼琅诸苎纂硅始嗨傥燃臂呆贵屹壮亍蚀卅豹邬迭浊童螂捐圩勐触寞壤荫膺渌芳懿遴螈泰蓼蛤茜舅枫朔膝眙避梅判鹜璜藻黔侥懂腈札丞慈顿摹荻琬斧沈滂胁莜匀鄄掌绰茎焚赋萱谑汁铒瞎夺野娆冀弯篁懵灞隽芡俐辩芯喏觐悚蔗熠鼠呵橼峨畜缔禾崭弃熊凸拗穹蒙抒劝闫扳阵踪喵侣搬仅荧蝾琦买婧瞄寓皎冻赝箩莫瞰郊姝筒枪煸袋舆痱涛母启践绲盘遂昊槿纰泓惨檬越Co憩熵祷钒暧塔阗咄魔钞邻扬杉殴咽弓吭揽霆殖脆彻岩芝勃辣剌嘎甄佘皖伦授徕挪皇庞稔芜踏兖卒擢鳞颗斯捧琮讹蛙纽谭酸兔莒睇伟觑羲宜褐旎辛卦诘筋鎏溪挛熔阜晰鳅丢奚灸呱献黛鸪萨拯洲辑叙恻谒允柔烂氏漆拎惋扈湟纭啕掬哥忽涤靡郗瓷扁廊雏钮敦懦汀啉岸瞅尊眩飙忌仝迦熬毫胯茄舛锵诧羯後漏汤宓仞蚁壶谰皑铄罔辅晶苦牟闽烃饮聿丙朱煤涔鳖犁罐荼淦妤戎孑婕瑾戢钵枣砥桠稣阎肃梏孪昶衫嗔侃塞蜃樵峒貌屿阐栖珞荭吝萍恂啻磬峋俸豫谎镍韬魇晴U囟猜蛮坐伴亭肝佗蝠妃胞滩榴氖垩苋砣扪馏姓轩厉侈禀垒岑赏钛辐披荥沅悔铧帼蝇ayng哀浆瑶凿桶皮奴苜佤伶晗铱炬优氢恃甫端锌灰稹曙亥碾拉萝绔捷浍姑菖凌涞麽桨潢绎镰锑渝铬困绽觎匈糙暑裹鸟迷綦亳佝骥仆婶郯瀹脖踞针晾忒瞩叛椒邗肆玫忡咧唆潦笛阚沸菽贫镂麝鸾衬恪叠希粤爻喝茫郸庸碟妹膛叮饵崛嗲椅搅咕敛尹垦闷蝉霎勰败泸肤焦浠鞍刁舰乙竿裔茵函伊兄娜謇莪宥似蝽翳翠粑薇祢骏赠叫竖芗莠潭俊羿耜O趁嗪囚芒洁笋鹑硝堡渲揩携宿遒颍棱葵琴捂饰衙耿岂涟蔺瘤柞怜匹炜哆秦缎幼茁绪恨楸娅瓦桩雪嬴伏榔妥铿拌眠雍缇卓屈哧咦巅娑侑淫膳祝勾姊胄疃薛蜷巷芙芋熙闰勿剩钏幢陟铛慧浙浇飨惟绗祜澈啼咪磷诅郦抹跃壬吕肖琏剡赚泊津宕殷氲漫邺涎怠遵俏叹孙筵鞭箭c眸祭髯啖坳愁芩倡巽穰沃胚怒凤槛剂趵邢灯鄢桐睽锯槟婷嵋圻诗蕈颠芸馥竭锗徜恭籁剑苡龄僧桑潸弘澶楹悲讫悸谍椹呢桓葭攫翰躲敖柑郎笨魁燎葩磋垛玺狮沓砜蕊锺蕉翱闾巫旦茱嬷枯鹏贡芹汛矫禺佃舫惯趋疲挽岚虾衾飓铖孩稞瑜壅勘妓畅髋庐牲蓿榕练垣唱邸菲昆穿绡麒愚泷涪漳妩娉榄讷觅旧藤煮柳叭庵烷阡罂猖咿媲脉貅黠熏哲烁坦兜潇撒珩圹乾摸樟帽襄魂轿憬锡喃皆咖隅脸残泮鹂珊囤咤误闹淙芊淋囗拨梳渤R绨婀幡狩麾谢旌伉纶裂驳砼咛澄樨蹈宙澍倍貔操勇蟠摈砧虬够缁悦藿摁淹豇虎榭吱d喧荀奋偕犍惮坑璎宛妆袈倩窦昂荏乖怅撰牙袁琼雁趾荚虻涝杏韭偈绫鞘卉蓥杭荨匆竣簪辙敕虞丹缭黟m淤瑕铉硼茨嶂畸敬涿粪窘熟叔嫔盾忱裘梵赡珙咯娘庙溯胺葱摊荷卞乒髦寐铭胗枷爆溟羚轨惊罄竽菏氧浅楣盼枢炸阆杯谏淇渺俪秆泪跻渡耽釜鳎煞呗韶舶鹳缜旷皱檀霖奄槐艳蝶旋骞腊盈丁蜚矸蝙睨僻鬼醴夜彝磊笔拔栀糕厦邰纫纤膊躇烯蘼冬诤暄骶哑丕愈咱螺跋搏笠淡骅谧鼎皋姚蠢驼耳涯狗蒽孓犷凉芦箴铤孤坤V茴朦挞尖橙诞搴洵浚漯柘嚎讽芭咻祠秉跖埃吓糯眷馒惹娼鲑嫩讴轮靶褚缤宋帧删驱碎扑俄涣竹噱皙佰渚斡镉刀崎筐佣夭贰肴峙艿匐牺镛缘仡嫡劣枸堀梨簿鸭蒸亦稽浴衢束阁棋潋聪睛插冉阪苍搽蟾幸仇樽撂慢幔淅覃觊溶妖帛侨曰泗殳玥佥盧谞仵銘璠蒯羅陳禎龍厶學钭葉楊諭旸喆畛铫珺湧鎔肜奡隰厍赟枰澂棂峇蘅瘳旻穂鐘達岍焓芃屺昇呂鋆韻捃潔黃琤鼐琍刖幨劉瑄鄭瑋怿韡鹘淩粢贠綺鳳俶琀媪焜維纥瓘撄勳逦粞淏鴻聖浛枘欸骉繻骀恫趙酆钤翀鼗祎颀杼戆垚慶樯張穎僮镕鍙姞璘豐乂傑柢苠髟鲮苾沚沄蕭輝軒鄍謦蓊晞嫄艽莛贽珦珮莩蘇寧筴洹禚玙欷叆夢馮曌舢瀞栊雩袆亞旵炆興瑭錢華骈彥誠琇彧賴鄧偉謝忾氷嫱啟郤禤苻鈕倫翮鹪嶷瑀昳樂喬鲭郇滉蓮苌谳甦酩囝铼贇姫赙蓁斫玒玍鈺吳虿瑩貞晅槱帻桤溫儌珽轹璟贶榇聃漢蹀蔹咲燊傧揚玘骢許忝箸畑畋珅邘烜狻猊暘伭涴躅埝缑冴囯锿扆颙麗溦爰尠旼偲潆珣孫埏谖繇犇蠼柽禛權馕舂怊揆蔣鲋筆邠菹堃镔筲珪玚珝賢鋹嬛燏燚镥杓塄駉暕镡洳勖樸玢笳袞蒈遆東顼闡跞妺訔垲辇昉葶窳準雲镪傪蛩鈞棁婍菥趑蕻氾伲柃昪荈嬌競镈媵禳隈倞翃俵幵阼靜闞钲馂埴琭礞燨聍腚緯荑璆蘋凊濛佾劼馑鹛檄蘭眇襦赆俦堯郄嫚俣叡與锞歡鹕妘逖仉皊絜槔玦兕砫環觌騄廪汯闿荜弢埜锒彐愔礽郐奭辂駱珧璿虼勹桄連剀熳竑骝墁颥忞詠菘玊锾馧僖钫坜吣夤凃錤劢絵犴紹剛嫲曈昃鲡怩堾谶煳馬軍茑綠骒玓佀藍矍黉薷嶽慊晊堅純辔麈纮祐語齊樗筻偄荇峯錞旃悛笾圄镵睎芨芑镫憲樺滈萩闩梣罟縠韪锖綝禇嫘勮儀姖俤溁葸埸薜缌钶媞墀杪滏閔愛筌柙鋈發雎鵾倢铷財沖虓稆琹嫠粦埮強霈柝哿岵讧沔圉卬瑢蒗锬鏻堋瑨長曉暐訚婥崑姮蘩貴邤鍾榮刭榀阌暌鹮沆詩棹梶噙蔌毬夼骎璮槃蒨勔挢棼铴妸锃抟茳査亶茈昺遠澣飑缲來湋蚋瞢芄崟凱叚琋阏樑堞禥雫皛昰澪啷円汜谠皦觫齼鳘酾栢癡跗訝葙晛愷梃魃蒴鸰轳懔倧韋頔磔笪悫堠犛巖沨囏岄黦锘龢搡琎蓱鸷簡峘劓铳颎伷頫颋順尨禮佁蘧帔皞盍眧嬼鬈呙薮彖硎逋徭恚缣穇龀馝蒺艟渫翥佺蜮玹伋隤遘锩霑錡帾婳桕楗卍弇覞玎纡黥炤砉鹍璕岓駿銎枨钪眘沇訢簙鰆挹涼榊诮玠廙昚洺攉匄燠妁钋汭湑俅莶骘顏嗌伧丌槊嬖媸蜞龕鲊翕槁轾珒墼洧沩辚泆義変忺缟赀楮僡锜龑煚镏勝晉綸岽螓谫僾桫濬鹋暲尐圔瓊圯瑱欑倕欤怛阋鞒驺昈傳翾韓菂钘昴逶洸霨婼冏聶黾仂胨躞篦嵛喾瑠廑琯瑤酃杬攮缵缳浡筇溍辻赧瓅宬訏洣芏恝毳蠲洎児镲洇澌枥浞谵傜檦疴耔鳏锱佴寔弨繄陽茔婞琫協飏錾芈荛捱儇凼曽枧镘棽嬿妣箅甑鐾樘钹邨铙崚魈煕钅祾謙汎彊焴柊檫鬯鷟讬逴穑惝萬湉伡嵚臌迨泚荌菔喑柁誌伃撙漼镠犨婨僎玼苧國蓠卨遑澴圊棻龆詝揿稂邊烺畯楫泐轸瑸驌喈茇忪徂儁鍊塨嗥偓萏钐姁鸉暾樋姵瞋葚谂诔舁箎閤垧珷蒉頲龠竦艨瑧莢枬踅寶饔弁甡堉舣熇姼陸絪獬鳴迴檎揶澥槑茀嵓惇俆諴嘏蕖鎗莙锓衠懽哙忸覌槙躜﻿燦撖遊騉迵檑蓰垕皝檠尢嵐藥锠缋顧岬笞珖穊綵偘倮旆邾唪枊粝睍憶隹畊硌浯蛘孅攽玕慜廼牮増睆锫嫜苈歂復鋐敩嫪毐踕孖霙抃盌媖薨逡炘睕術皭崿悊敉毖適廣鷩鸶悝槻栌缶昞蕫歩雝譞瘐皕洢顒茕俍臁阇願薫赜龐劻俜談烔邙劦紬皪潓爔瓛嶌嵝勍籴獾栄關錱洌讚癀垤熘湣儦卮繡镟仳郃砵鄠裼戥別卣嘭赇躭鱚澔傢經缯崞鳇爿鎮焐斉皤宼佊斎篥愍孀暋衎洑缗勣鄔蚺羕憕亜賁笹懃怍窬蚡袛艤徳秾斅髙燔緤牕紞鲕圮炓椠媌彡笥熜鎧書轫驵镞眆蓂柷钌炷芺狷皥匽摭菸鈡諺旳裎酺钍锎麇鸧曛塏茺炟浬鹞坫泖暠搠訇窀荦匏鼷湦戅禕碹侪昫脩嵒肫膑莤麥嵫啓頋妏伬湲捰娒嬋榘傼胐厓橐劬湯枏趱臬绹煏慆讦轺曩嬅琲汔鉑廆脔鍌杕進枵闶瞽癯朓壸冱薗裟鹩燄旒硏鹚稲祧翛赑骕骦箦糸谿镩嶕俙尥諹鲒垇玭棐臞芶峬棪熛痖霂棡鹈麴礎韂湜忤鲼瑔騣鷲凫爝溵菑荘鸫鹇袪蒹鑾勠笮彳俈醮蝻顗翚蹻跬倖幛澐肸蕓鼹屾繆桝愠罃伢賈儆黧弌潾黹嵕誊鬱舄闕玸蝓骍嗄戭涫淯栯選焮頠豳莊碏羼鏊舲睃驩咝渼髡釐缡覈隨剟玪徹霪伥佷醣魑侔蔀鋕蟲鈊喦瑒筰鍟瘅娖蒒簟範踺坌纩侂愊倓漈眰欻琂垆彙貑漦缊汆堄傺饒佈囵孥瀍鑑関趼嵴驎麿磙帙唎偁隣搷孮棓橥桊燿黇苊勗黪嵎庯屘稘穀攵珰赒衒魯乇湙滠倉尋郞棲匋孃梼荍蕗戡棨葧皀憷砱晫卐颿棄蠙郉峣秄晙矞匷鲾俳锽螽堍镨鄜畠躍諲畢麹盉倅矦簖崙楩媗跸敱瓈磾萑竸忭枹槠孱褆彘迮簃锳縻旴扃疋勅塚闼垍襞赪聽煃镤迺殄訸冨璥鞏璗蝏嬤艈奀瑮毹巻茚媦蜢眛薬晹瓎続钔豨薿璝罅窋闇箪暏獸渟鈇埒蒇嫫髑綮蛉浰沲聩蕰祓濆姍'
    recognition_list = []
    for i in range(len(regions_list)):
        temp_pic = img[regions_list[i][1]:regions_list[i][3]+1,regions_list[i][0]:regions_list[i][2]+1,:]
        temp_pic = temp_pic[np.newaxis, :, :, :]
        recognition_result = sess.run(decoded_final, feed_dict={x: temp_pic, is_train: np.array([1]), seq_length: np.array([24])})
        final = ''
        for i in recognition_result[0][0]:
            final += label[i]
        recognition_list.append(final)

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