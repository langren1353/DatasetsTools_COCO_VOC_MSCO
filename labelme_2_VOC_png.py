import os
import glob

base_path = 'C:\\Users\\AC\\Desktop\\train\\'  # TODO path是你存放json的路径

def gen_png_jsondir():
    '''
    :return: 传入base_path, 然后直接生成对应的mask.png和一些yaml结果数据
    '''
    jsonlist = glob.glob(base_path + "*.json")
    for filepath in jsonlist:
        print("处理："+filepath)
        os.system("python D:\Anaconda3\Scripts\labelme_json_to_dataset.exe " + filepath)

def copy_png_toDir(toDir):
    # TODO 未完成项目
    pass