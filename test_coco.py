# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 14:38
# @Author  : zhoujun

from pycocotools.coco import COCO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

annFile='./ICDAR2015_train.json'
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# imgIds = coco.getImgIds(imgIds = [324158])
imgIds = coco.getImgIds()
imgId=np.random.randint(0,len(imgIds))
img_info = coco.loadImgs(imgIds[imgId])[0]
dataDir = r'E:\zj\dataset\icdar2015\train\img'

img_show = plt.imread('%s/%s' % (dataDir, img_info['file_name']))




# load and display instance annotations
# 加载实例掩膜
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# catIds=coco.getCatIds()
catIds=[]
for ann in coco.dataset['annotations']:
    if ann['image_id']==imgIds[imgId]:
        catIds.append(ann['category_id'])

annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)

plt.axis('off')
plt.imshow(img_show)
coco.showAnns(anns)

plt.show()
