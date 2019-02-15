# coding=utf-8

import os
import pickle
import cv2
import shutil
import random
from collections import defaultdict, OrderedDict
from tqdm import tqdm


def process2model_color(root):
    """
    处理所有标注有model和color数据
    返回ID2imgs
    """

    # 求model_attr.txt和color_attr.txt中ID交集
    model_attr_txt = root + '/attribute/model_attr.txt'
    color_attr_txt = root + '/attribute/color_attr.txt'
    img2vid_txt = root + '/attribute/img2vid.txt'

    assert os.path.isfile(model_attr_txt) and \
        os.path.isfile(color_attr_txt)

    model_ids, color_ids = set(), set()
    with open(model_attr_txt, 'r', encoding='utf-8') as f_h_model, \
            open(color_attr_txt, 'r', encoding='utf-8') as f_h_color:
        for line in f_h_model.readlines():
            model_ids.add(int(line.strip().split()[0]))
        for line in f_h_color.readlines():
            color_ids.add(int(line.strip().split()[0]))

    # get intersection
    MC_IDs = list(model_ids & color_ids)
    MC_IDs.sort()
    # print(MC_IDs)
    print('=> toal %d vehicle IDs with model and color labeled.' % len(MC_IDs))

    # class ID mapping: vid2TrainID <=> trainID2Vid
    vid2TrainID, trainID2Vid = defaultdict(int), defaultdict(int)
    for k, vid in enumerate(MC_IDs):
        vid2TrainID[vid] = k
        trainID2Vid[k] = vid

    # statistics of ID2img_list
    if os.path.isfile(img2vid_txt):
        All_IDs = set()
        ID2imgs = defaultdict(list)

        with open(img2vid_txt, 'r', encoding='utf-8') as f_h:
            sample_cnt = 0
            for line in f_h.readlines():
                line = line.strip().split()

                img, id = line

                All_IDs.add(id)
                ID2imgs[int(id)].append(img)
                sample_cnt += 1

    # print(ID2Num)
    print('=> total %d vehicles have total %d IDs' %
          (sample_cnt, len(All_IDs)))

    # 序列化MC_IDs和ID2imgs到attribute子目录
    MC_IDS_path = root + '/attribute/MC_IDs.pkl'
    ID2imgs_path = root + '/attribute/ID2imgs.pkl'
    # ID2imgs = sorted(ID2imgs.items(),
    #                  key=lambda x: int(x[0]),
    #                  reverse=False)
    print(len(ID2imgs))
    with open(ID2imgs_path, 'wb') as f_h_1, \
            open(MC_IDS_path, 'wb') as f_h_2:
        pickle.dump(ID2imgs, f_h_1)
        pickle.dump(MC_IDs, f_h_2)
        print('=> %s dumped.' % ID2imgs_path)
        print('=> %s dumped.' % MC_IDS_path)

    # 序列化用于model,color的Vehicle mapping
    vid2TrainID_path = root + '/attribute/vid2TrainID.pkl'
    trainID2Vid_path = root + '/attribute/trainID2Vid.pkl'
    with open(vid2TrainID_path, 'wb') as f_h_1, \
            open(trainID2Vid_path, 'wb') as f_h_2:
        pickle.dump(vid2TrainID, f_h_1)
        pickle.dump(trainID2Vid, f_h_2)
        print('=> %s dumped.' % vid2TrainID_path)
        print('=> %s dumped.' % trainID2Vid_path)


# model, color: multi-label classification
def split2MC(root,
             TH=0.1):
    """
    根据ID2imgs和MC_IDS划分到新目录
    @TODO: 还需要对生成的数据集进行可视化验证
    """

    mc_ids_f_path = root + '/attribute/MC_IDs.pkl'
    id2imgs_f_path = root + '/attribute/ID2imgs.pkl'
    model_attr_txt = root + '/attribute/model_attr.txt'
    color_attr_txt = root + '/attribute/color_attr.txt'

    assert os.path.isfile(mc_ids_f_path) \
        and os.path.isfile(id2imgs_f_path) \
        and os.path.isfile(model_attr_txt) \
        and os.path.isfile(color_attr_txt)

    # 读取veh2model和veh2color
    vid2mid, vid2cid, img2vid = defaultdict(
        int), defaultdict(int), defaultdict(int)
    with open(model_attr_txt, 'r', encoding='utf-8') as fh_1, \
            open(color_attr_txt, 'r', encoding='utf-8') as fh_2:
        for line in fh_1.readlines():  # vid to model id
            line = line.strip().split()
            vid2mid[int(line[0])] = int(line[1])
        for line in fh_2.readlines():  # vid to color id
            line = line.strip().split()
            vid2cid[int(line[0])] = int(line[1])

    with open(mc_ids_f_path, 'rb') as f_h_1, \
            open(id2imgs_f_path, 'rb') as f_h_2:
        mc_ids = pickle.load(f_h_1)
        id2imgs = pickle.load(f_h_2)

        train_txt = root + '/attribute/train_all.txt'
        test_txt = root + '/attribute/test_all.txt'

        # 按照Vehicle ID的顺序生成训练和测试数据
        train_cnt, test_cnt = 0, 0
        with open(train_txt, 'w', encoding='utf-8') as f_h_3, \
                open(test_txt, 'w', encoding='utf-8') as f_h_4:
            for i, vid in enumerate(mc_ids):
                if vid in mc_ids:
                    imgs_list = id2imgs[vid]
                    for img in imgs_list:
                        # get image and label: img + model_id + color_id
                        model_id, color_id = vid2mid[vid], vid2cid[vid]
                        img_label = img + ' ' + str(model_id) \
                            + ' ' + str(color_id) + ' ' + str(vid) + '\n'

                        # split to train.txt and test.txt
                        if random.random() > TH:
                            f_h_3.write(img_label)
                            train_cnt += 1
                        else:
                            f_h_4.write(img_label)
                            test_cnt += 1
                    # print('=> Vehicle ID %d samples generated.' % vid)

            print('=> %d img files splitted to train set' % train_cnt)
            print('=> %d img files splitted to test set' % test_cnt)
            print('=> total %d img files in root dataset.' %
                  (train_cnt + test_cnt))


def process_vehicleID(root, TH=15):
    """
    统计VehicleID ID数
    """
    # 遍历所有图片
    img2vid_f_path = root + '/attribute/img2vid.txt'
    if os.path.isfile(img2vid_f_path):
        IDs = set()
        ID2imgs = defaultdict(list)

        with open(img2vid_f_path, 'r', encoding='utf-8') as f_h:
            sample_cnt = 0
            for line in f_h.readlines():
                line = line.strip().split()

                img, id = line

                IDs.add(id)
                ID2imgs[id].append(img)
                sample_cnt += 1

    # print(ID2Num)
    print('=> total %d vehicles have total %d IDs' % (sample_cnt, len(IDs)))

    # 序列化满足条件的Vehicle IDS
    ID2imgs_path = root + '/attribute/ID2imgs.pkl'
    ID2imgs = sorted(ID2imgs.items(),
                     key=lambda x: int(x[0]),
                     reverse=False)

    # print(ID2imgs)
    ID2imgs_sort = defaultdict(list)
    for item in ID2imgs:
        if len(item[1]) >= TH:
            ID2imgs_sort[item[0]] = item[1]
    print('=> total %d Ids meet requirements' % len(ID2imgs_sort.keys()))

    # print(ID2imgs_sort)
    print('=> Last 10 vehicle ids: ', list(ID2imgs_sort.keys())[-10:])

    with open(ID2imgs_path, 'wb') as f_h:
        pickle.dump(ID2imgs_sort, f_h)
        print('=> %s dumped.' % ID2imgs_path)

    # 验证筛选结果...
    fetch_from_vechicle(ID2imgs_path, root, img)


def fetch_from_vechicle(ID2imgs_path, root, img):
    # 验证筛选结果...
    print('=> testing...')
    with open(ID2imgs_path, 'rb') as f_h:
        ID2imgs = pickle.load(f_h)

        img_root = root + '/image'
        dst_root = 'f:/VehicleID_Part'

        # 按子目录存放
        for i, (k, v) in enumerate(ID2imgs.items()):
            id, imgs = k, v
            imgs = [img_root + '/' + img + '.jpg' for img in imgs]
            # print(imgs)

            dst_sub_dir = dst_root + '/' + str(i)
            if not os.path.isdir(dst_sub_dir):
                os.makedirs(dst_sub_dir)

            # copy to test_result
            for img in imgs:
                shutil.copy(img, dst_sub_dir)
            print('=> %s processed.' % dst_sub_dir)


def get_ext_files(root, ext, f_list):
    """
    递归搜索指定文件
    :param root:
    :param ext:
    :param f_list:
    :return:
    """
    for x in os.listdir(root):
        x_path = root + '/' + x
        if os.path.isfile(x_path) and x_path.endswith(ext):
            f_list.append(x_path)
        elif os.path.isdir(x_path):
            get_ext_files(x_path, ext, f_list)


def split(data_root,
          RATIO=0.1):
    """
    将按照子目录存放的数据划分为训练数据集和测试数据集
    """
    if not os.path.isdir(data_root):
        print('=> invalid data root.')
        return

    train_txt = data_root + '/train.txt'
    test_txt = data_root + '/test.txt'

    # 写train.txt, test.txt
    train_cnt, test_cnt = 0, 0
    with open(train_txt, 'w', encoding='utf-8') as f_train, \
            open(test_txt, 'w', encoding='utf-8') as f_test:
        # 从根目录遍历每一个子目录
        sub_dirs = [sub for sub
                    in os.listdir(data_root) if sub.isdigit()]
        sub_dirs.sort(key=lambda x: int(x))
        for sub in sub_dirs:
            sub_path = data_root + '/' + sub
            if os.path.isdir(sub_path):
                for img in os.listdir(sub_path):
                    # 写txt文件
                    img_path = sub_path + '/' + img
                    relative_apth = '/'.join(img_path.split('/')[2:])
                    if random.random() > RATIO:
                        f_train.write(relative_apth + '\n')
                        train_cnt += 1
                    else:
                        f_test.write(relative_apth + '\n')
                        test_cnt += 1
        print('=> %d img files splitted to train set' % train_cnt)
        print('=> %d img files splitted to test set' % test_cnt)
        print('=> total %d img files in root dataset.' %
              (train_cnt + test_cnt))


def form_cls_name(root):
    """
    加载类别和类别序号的映射,
    序列化到attribute目录
    """
    model_names_txt = root + '/attribute/model_names.txt'
    color_names_txt = root + '/attribute/color_names.txt'
    if not (os.path.isfile(model_names_txt) and
            os.path.isfile(color_names_txt)):
        print('=> [Err]: invalid class names file.')
        return

    modelID2name, colorID2name = defaultdict(str), defaultdict(str)
    with open(model_names_txt, 'r', encoding='utf-8') as fh_1, \
            open(color_names_txt, 'r', encoding='utf-8') as fh_2:
        for line in fh_1.readlines():
            line = line.strip().split()
            modelID2name[int(line[1])] = line[0]

        for line in fh_2.readlines():
            line = line.strip().split()
            colorID2name[int(line[1])] = line[0]
    print(modelID2name)
    print(colorID2name)

    # 序列化到硬盘
    modelID2name_path = root + '/attribute/modelID2name.pkl'
    colorID2name_path = root + '/attribute/colorID2name.pkl'
    with open(modelID2name_path, 'wb') as fh_1, \
            open(colorID2name_path, 'wb') as fh_2:
        pickle.dump(modelID2name, fh_1)
        pickle.dump(colorID2name, fh_2)

    print('=> %s dumped.' % modelID2name_path)
    print('=> %s dumped.' % colorID2name_path)


# ----------------- 从10086之外取test-pairs数据用来测试
def gen_test_pairs(root):
    """
    """
    

if __name__ == '__main__':
    # process_vehicleID(root='e:/VehicleID_V1.0')
    # split(data_root='f:/VehicleID_Part')

    process2model_color(root='e:/VehicleID_V1.0')
    split2MC(root='e:/VehicleID_V1.0',
             TH=0.05)

    # form_cls_name(root='e:/VehicleID_V1.0')

    print('=> Done.')
