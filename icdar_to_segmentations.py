from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import shutil
import re
import progressbar
import argparse
import time
from multiprocessing import pool

parser = argparse.ArgumentParser()
parser.add_argument('--input-image', default='/dockerdata/ic19/train_image/')
parser.add_argument('--input-gt', default='/dockerdata/ic19/train_gt/')
parser.add_argument('--output-image', default='icdar/train2019/')
parser.add_argument('--output-anno', default='icdar/annotations')
parser.add_argument('--max', type=int, default=30)
args = parser.parse_args()

INPUT_IMAGE_PATH = Path(args.input_image)
INPUT_GT_PATH = Path(args.input_gt)
OUTPUT_IMAGE_PATH = Path(args.output_image)
OUTPUT_ANNO_PATH = Path(args.output_anno)


class TextBox:
    def __init__(self, line):
        splits = line.strip().split(',')
        self.coors = list(map(int, splits[:8]))
        self.ignore = splits[-1] == '###'

    def show(self):
        print(self.coors, self.ignore)


def load_annotations(file_path):
    boxes = []
    with file_path.open('r', encoding='utf-8-sig') as f:
        for line in f:
            boxes.append(TextBox(line))
    return boxes


def main():
    r_id = re.compile(r'\d+')
    image_list = [f for f in INPUT_IMAGE_PATH.iterdir()]
    if args.max:
        image_list = image_list[:args.max]
    start_time = time.time()
    for im_file in progressbar.progressbar(image_list):
        gt_file = INPUT_GT_PATH / im_file.with_suffix('.txt').name
        # 如果找不到对应的 gt 文件，则跳过这张图片
        if not gt_file.exists():
            continue
        im_id = int(r_id.search(im_file.name).group())
        new_im_file = OUTPUT_IMAGE_PATH / Path(f"{im_id}.jpg")
        im = Image.open(im_file)
        # 如果图片的色彩通道以及后缀都符合要求，则直接复制
        if im.format == 'RGB' and im_file.suffix == '.jpg':
            shutil.copyfile(im_file, new_im_file)
        # 否则的话，先转换
        else:
            im = im.convert('RGB')
            im.save(new_im_file)
        # 准备开始绘制 mask，先准备一个纯黑的画布
        black_canvas = Image.new('RGBA', im.size, (0, 0, 0))

        def draw_text_mask(box, index):
            canvas = black_canvas.copy()
            draw = ImageDraw.Draw(canvas)
            draw.polygon(box.coors, fill=(255, 255, 255))
            seg = OUTPUT_ANNO_PATH / Path(f'{im_id}_text_{index}.png')
            canvas.save(seg)
            return canvas

        text_masks = []
        boxes = load_annotations(gt_file)
        # 多线程
        mypool = pool.Pool(processes=4)
        for index, box in enumerate(boxes):
            mypool.apply_async(draw_text_mask, args=(
                box, index), callback=text_masks.append)
        mypool.close()
        mypool.join()
        # 单线程
        # for index, box in enumerate(boxes):
        #     text_masks.append(draw_text_mask(box, index))
        # 绘制纯背景图层
        canvas = np.asarray(black_canvas).astype(np.int32)
        for mask in text_masks:
            canvas += np.asarray(mask)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        canvas[:, :, :3] = 255 - canvas[:, :, :3]
        canvas = Image.fromarray(canvas)

        o = OUTPUT_ANNO_PATH / Path(f'{im_id}_background_0.png')
        canvas.save(o)
    stop_time = time.time()
    print(f'Elapsed time is {stop_time - start_time:.3f}s.')


if __name__ == "__main__":
    main()
