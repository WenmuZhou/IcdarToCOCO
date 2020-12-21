# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 12:53
# @Author  : zhoujun

import os
import numpy as np
import cv2
from tqdm import tqdm
from natsort import natsorted

from coco_annotation import CocoAnnotationClass


def get_polygons_from_annotation(annot_path):
    polygons = []
    with open(annot_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # remove weird encoding in start of gt file (train set)
        lines[0] = lines[0].replace("\ufeff", "")
        for l in lines:
            poly = l.split(",")[:8]
            poly = np.array(poly, dtype=np.int32).reshape((4, 2))
            polygons.append(poly)
    return polygons


def get_keyspoints(poly):
    kes = []  # 10*3 ,第一个是(x_mean.y_mean,match_type)
    # 中间四个是((x_i + x_mean) / 2 ,y_mean,match_type)
    # 后面四个是(x_mean,(y_i + y_mean) / 2 ,match_type)
    # 最后一个是((对角线交点x ,对角线交点y,match_type)
    match_type = mtype(poly)
    x_list = poly[:, 0]
    y_list = poly[:, 1]
    x_mean = x_list.mean()
    y_mean = y_list.mean()
    x_sort = sorted(x_list)
    y_sort = sorted(y_list)
    kes.extend((x_mean, y_mean, match_type))
    for x in x_sort:
        kes.extend(((x + x_mean) / 2, y_mean, match_type))
    for y in y_sort:
        kes.extend((x_mean, (y + y_mean) / 2, match_type))
    cpts = cross_point(poly)
    kes.extend((cpts[0], cpts[1], match_type))
    return kes


def cross_point(poly):
    from shapely.geometry import LineString
    from shapely.geometry.point import Point
    idx_list = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]
    cpts = []
    for idx in idx_list:
        line1 = LineString(poly[idx[:2]])
        line2 = LineString(poly[idx[2:]])
        cpts = line1.intersection(line2)
        if isinstance(cpts, Point):
            return [cpts.x, cpts.y]
    return cpts


def mtype(poly):
    from itertools import permutations
    from collections import defaultdict
    all_m_types = list(permutations(range(4), 4))  # 排列组合A44
    y_dict = defaultdict(list)
    for i, y in enumerate(poly[:, 1]):
        y_dict[y].append(i)
    poly = poly[poly[:, 0].argsort()]
    match = []
    for y in poly[:, 1]:
        match.append(y_dict[y].pop(0))
    return all_m_types.index(tuple(match))


def create_coco(data_list, phase, dataset_name, VIS=False):
    coco_annot = CocoAnnotationClass(['text'], "icdar15_ist")  # COCO IS 1-indexed, don't include BG CLASS
    ANNOT_ID = 0
    for i, (img_path, gt_path) in enumerate(tqdm(data_list)):
        img_name = os.path.basename(img_path)
        img_id = i#img_name.split('.')[0].replace('img_','')
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read %s" % (img_path))
            continue

        img_height, img_width = img.shape[:2]

        if not os.path.exists(gt_path):
            print("Could not find %s" % (gt_path))
            continue

        polys = get_polygons_from_annotation(gt_path)

        for poly in polys:
            if len(poly) < 3:
                continue
            ANNOT_ID += 1
            keypoints = get_keyspoints(poly)
            coco_annot.add_annot(id=ANNOT_ID, img_id=img_id, img_cls=1, seg_data=poly.astype(np.float32),
                                 keypoints=keypoints)
            if VIS:
                n = len(poly)
                for ix, px in enumerate(poly):
                    px = tuple(px)
                    cv2.line(img, px, tuple(poly[(ix + 1) % n]), (0, 255, 0))
                    cv2.circle(img, px, 2, (0, 0, 255), -1)
        if VIS:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        coco_annot.add_image(id=img_id, width=img_width, height=img_height, file_name=img_name)
    coco_annot.save("{}_{}.json".format(dataset_name, phase))


def get_data_list(file_name):
    with open(file_name, encoding='utf8') as f:
        return natsorted([x.replace('\n', '').split('\t') for x in f.readlines()])


if __name__ == '__main__':
    img_dir = r"E:\zj\dataset\icdar2015\test\img"
    annot_dir = r"E:\zj\dataset\icdar2015\test\gt"

    data_list = get_data_list(r'E:\zj\dataset\icdar2015\test\test.txt')
    create_coco(data_list, 'test', dataset_name='ICDAR2015')
