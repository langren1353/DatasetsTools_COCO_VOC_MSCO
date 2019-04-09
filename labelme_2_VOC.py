#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
import PIL.Image

import labelme

'''
python labelme_2_VOC.py 当前 当前/out
'''

# 对应id 0  1  2
class_names = ("_background_", "Building", "Water")
class_name_to_id = {class_names[i]:i for i in range(len(class_names))}
print(class_name_to_id)
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inDir')
    parser.add_argument('outDir')
    args = parser.parse_args()

    if osp.exists(args.outDir):
        print('Output directory already exists:', args.outDir)
        quit(1)
    os.makedirs(args.outDir)
    os.makedirs(osp.join(args.outDir, 'JPEGImages'))
    os.makedirs(osp.join(args.outDir, 'SegmentationClass'))
    os.makedirs(osp.join(args.outDir, 'SegmentationClassVisualization'))
    print('Creating dataset:', args.outDir)

    print('class_names:', class_names)
    out_class_names_file = osp.join(args.outDir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelme.utils.label_colormap(255)

    for label_file in glob.glob(osp.join(args.inDir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                args.outDir, 'JPEGImages', base + '.jpg')
            out_lbl_file = osp.join(
                args.outDir, 'SegmentationClass', base + '.npy')
            out_viz_file = osp.join(
                args.outDir, 'SegmentationClassVisualization', base + '.jpg')

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))
            PIL.Image.fromarray(img).save(out_img_file)

            lbl = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )

            # Only works with uint8 label
            # lbl_pil = PIL.Image.fromarray(lbl, mode='P')
            # lbl_pil.putpalette((colormap * 255).flatten())
            np.save(out_lbl_file, lbl)

            label_names = ['%d: %s' % (cls_id, cls_name)
                           for cls_id, cls_name in enumerate(class_names)]
            viz = labelme.utils.draw_label(
                lbl, img, label_names, colormap=colormap)
            PIL.Image.fromarray(viz).save(out_viz_file)


if __name__ == '__main__':
    main()