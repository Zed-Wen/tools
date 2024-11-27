import zipfile
import os
import shutil

def getfiles(path):
    filenames=os.listdir(path)
    return filenames










dirname = '2022-09-18 74'
path = os.path.join(os.getcwd(),dirname)
files = getfiles(path)
print(files)

for file in files:
    print('processing' + file)
    # 压缩文件路径
    zip_path= os.path.join(path,file)

    # 删掉损坏的zip
    if os.path.getsize(zip_path) < 4000:
        print('Zip file is too small. This should be an invalid file.')
        os.remove(zip_path)
        continue

    # 文件存储路径
    label = file[:-4]
    save_path = os.path.join(os.getcwd(),"extract",label)

    # 读取压缩文件
    file=zipfile.ZipFile(zip_path)
    # 解压文件
    print('开始解压...')
    file.extractall(save_path)
    print('解压结束。')
    # 关闭文件流
    file.close()
    os.remove(zip_path)