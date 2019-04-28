import datetime
import json
from pathlib import Path
import re
from PIL import Image
import numpy as np
import progressbar
from multiprocessing import pool
from pycococreatortools import pycococreatortools

ROOT_DIR = Path('icdar')
IMAGE_DIR = ROOT_DIR / Path('train2019')
ANNOTATION_DIR = ROOT_DIR / Path('annotations')

INFO = {
    "description": "ICDAR 2019 MLT Dataset",
    "url": "http://rrc.cvc.uab.es/?ch=15",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "mikoto",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'text',
        'supercategory': 'shape',
    },
    {
        'id': 2,
        'name': 'background',
        'supercategory': 'shape',
    }
    # {
    #     'id': 3,
    #     'name': 'ignore',
    #     'supercategory': 'shape',
    # }
]


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    im_files = [f for f in IMAGE_DIR.iterdir()]
    im_files.sort(key=lambda f: int(f.stem))

    for im_file in progressbar.progressbar(im_files):
        image = Image.open(im_file)
        im_info = pycococreatortools.create_image_info(
            image_id, im_file.name, image.size
        )
        coco_output['images'].append(im_info)

        gt_files = [f for f in ANNOTATION_DIR.iterdir(
        ) if re.match(f'{segmentation_id}_', f.stem)]
        myPool = pool.Pool(processes=8)
        annotation_info_list = []
        for gt_file in gt_files:
            class_id = [x['id']
                        for x in CATEGORIES if x['name'] in gt_file.stem][0]

            category_info = {
                'id': class_id,
                'is_crowd': False  # 'background' in gt_file.stem
            }

            binary_mask = np.asarray(Image.open(
                gt_file).convert('1')).astype(np.uint8)
            myPool.apply_async(pycococreatortools.create_annotation_info, args=(
                segmentation_id, image_id, category_info, binary_mask, image.size
            ), callback=annotation_info_list.append)

            segmentation_id += 1

        myPool.close()
        myPool.join()
        for annotation_info in annotation_info_list:
            if annotation_info is not None:
                coco_output['annotations'].append(annotation_info)

        image_id += 1

    output_json = Path(f'instances_icdar_{IMAGE_DIR.stem}.json')
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(coco_output, f)


if __name__ == "__main__":
    main()
