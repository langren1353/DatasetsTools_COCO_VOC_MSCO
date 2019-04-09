# coding:utf-8

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab, os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
import random
from PIL import Image

'''
注意下面的参数都要修改，然后直接运行即可
注意VOC的数据全是灰度图，8位的图片
'''


dataDir = 'C:/Users/AC/Desktop'
CK5cats = ['Water', 'Building']
coco_train_path="C:/Users/AC/Desktop/train/train.json"
coco_val_path=  "C:/Users/AC/Desktop/val/val.json"


# 无需修改下面的值
CKdir = dataDir + "/" + "out"
CKimg_dir = CKdir + "/" + "images"
CKanno_dir = CKdir + "/" + "Annotations"
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


def mkr(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def showimg(coco, dataType, img, CK5Ids):
    global dataDir
    I = io.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()


def save_annotations(dataType, filename, objs):
    annopath = CKanno_dir + "/" + filename[:-3] + "xml"
    img_path = dataDir + "/" + dataType +"/" + filename
    dst_path = CKimg_dir + "/" + filename
    img = cv2.imread(img_path, 0)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    # shutil.copy(img_path, dst_path)
    cv2.imwrite(dst_path, img)
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(1) # 这里需要强制指定为1
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


def showbycv(coco, dataType, img, classes, CK5Ids):
    global dataDir
    filename = img['file_name']
    filepath = '%s/%s/%s' % (dataDir, dataType, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if name in CK5cats:
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = (int)(bbox[0])
                ymin = (int)(bbox[1])
                xmax = (int)(bbox[2] + bbox[0])
                ymax = (int)(bbox[3] + bbox[1])
                obj = [name, 1.0, xmin, ymin, xmax, ymax]
                objs.append(obj)
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(dataType, filename, objs)
    cv2.imshow("img", I)
    cv2.waitKey(1)


def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes

def handleannFile(type, ann_path):
    coco = COCO(ann_path)
    CK5Ids = coco.getCatIds(catNms=CK5cats)
    classes = catid2name(coco)
    for srccat in CK5cats:
        print(type + ":" + srccat)
        catIds = coco.getCatIds(catNms=[srccat])
        imgIds = coco.getImgIds(catIds=catIds)
        # imgIds=imgIds[0:100]
        for imgId in tqdm(imgIds):
            img = coco.loadImgs(imgId)[0]
            showbycv(coco, type, img, classes, CK5Ids)
            # showimg(coco,dataType,img,CK5Ids)

def get_CK5():
    mkr(CKimg_dir)
    mkr(CKanno_dir)

    handleannFile("train", coco_train_path)
    handleannFile("val", coco_val_path)


# split train and test for training
def split_traintest(trainratio=0.7, valratio=0.2, testratio=0.1):
    '''
    用于分割已有的样本集
    :param trainratio: 训练样本数量，70%
    :param valratio:   验证样本数量，20%
    :param testratio:  测试样本数量，10%
    :return: None
    '''
    dataset_dir = CKdir
    files = os.listdir(CKimg_dir)
    trains = []
    vals = []
    trainvals = []
    tests = []
    random.shuffle(files)
    for i in range(len(files)):
        filepath = CKimg_dir + "/" + files[i][:-3] + "jpg"
        if (i < trainratio * len(files)):
            trains.append(files[i])
            trainvals.append(files[i])
        elif i < (trainratio + valratio) * len(files):
            vals.append(files[i])
            trainvals.append(files[i])
        else:
            tests.append(files[i])
    # write txt files for yolo
    with open(dataset_dir + "/trainval.txt", "w")as f:
        for line in trainvals:
            line = CKimg_dir + "/" + line
            f.write(line + "\n")
    with open(dataset_dir + "/test.txt", "w") as f:
        for line in tests:
            line = CKimg_dir + "/" + line
            f.write(line + "\n")
    # write files for voc
    maindir = dataset_dir + "/" + "ImageSets/Segmentation"
    mkr(maindir)
    with open(maindir + "/train.txt", "w") as f:
        for line in trains:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/val.txt", "w") as f:
        for line in vals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/trainval.txt", "w") as f:
        for line in trainvals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/test.txt", "w") as f:
        for line in tests:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    print("spliting done")


if __name__ == "__main__":
    get_CK5()
    split_traintest()