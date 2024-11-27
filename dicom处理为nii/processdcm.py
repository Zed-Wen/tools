import SimpleITK as sitk
import tigre
import os
from xpinyin import Pinyin
import shutil


def check_and_creat_dir(fname):
    '''
    判断文件夹是否存在，不存在则创建文件夹
    '''
    if not os.path.exists(fname):
        os.makedirs(fname)
    else:
        return None


# 读取path下所有文件夹
def getfiles(path):
    filenames=os.listdir(path)
    return filenames

# 读取一个文件夹下的所有dcm并合成一个numpy
def read_dcm(path, folder_name):

    # 判断是否dicom文件过少。一般是1张到3张，这种一般是单张牙齿的；几十张的，这种一般是MR
    files = os.listdir(path)   # 读入文件夹
    num_dicom = len(files)       # 统计文件夹中的文件个数
    if num_dicom < 80:
        print('num of dicom is {}, will skip this folder.'.format(num_dicom))             # 打印文件个数
        return

    try:
        # 读取序列
        series_reader = sitk.ImageSeriesReader()
        seriesIDs = series_reader.GetGDCMSeriesIDs(path)
        fileNames = series_reader.GetGDCMSeriesFileNames(path, seriesIDs[2])
        series_reader.SetFileNames(fileNames)
        images = series_reader.Execute()

        # 转成numpy
        data_np = sitk.GetArrayFromImage(images)
        print(data_np.shape)


        if data_np.shape[0] < 80:
            print("Try again for seriesID3")
            fileNames = series_reader.GetGDCMSeriesFileNames(path, seriesIDs[3])
            series_reader.SetFileNames(fileNames)
            images = series_reader.Execute()
            # 转成numpy
            data_np = sitk.GetArrayFromImage(images)
            print(data_np.shape)

            if data_np.shape[0] < 80:
                print("Try again for seriesID4")
                fileNames = series_reader.GetGDCMSeriesFileNames(path, seriesIDs[4])
                series_reader.SetFileNames(fileNames)
                images = series_reader.Execute()
                # 转成numpy
                data_np = sitk.GetArrayFromImage(images)
                print(data_np.shape)

                if data_np.shape[0] < 80:
                    print("Try again for seriesID1")
                    fileNames = series_reader.GetGDCMSeriesFileNames(path, seriesIDs[1])
                    series_reader.SetFileNames(fileNames)
                    images = series_reader.Execute()      
                    # 转成numpy
                    data_np = sitk.GetArrayFromImage(images)
                    print(data_np.shape)

                    if data_np.shape[0] < 80:
                        print("Try again for seriesID0")
                        fileNames = series_reader.GetGDCMSeriesFileNames(path, seriesIDs[0])
                        series_reader.SetFileNames(fileNames)
                        images = series_reader.Execute()      
                        # 转成numpy
                        data_np = sitk.GetArrayFromImage(images)
                        print(data_np.shape)

                        if data_np.shape[0] < 80:
                            print('All index invalid shape. Will move to tohandcraft.')
                            shutil.move(path, os.path.join('tohandcraft',folder_name))
                            return None
    except IndexError:
        print('Find invalid index. Will move to tohandcraft.')
        shutil.move(path, os.path.join('tohandcraft',folder_name))
        return None

    # print(data_np.dtype)

    # plot
    # tigre.plotImg(data_np.transpose(1,2,0))

    ## save 
    # print(data_np.shape)
    # print(data_np.dtype)
    p = Pinyin()
    folder_name = p.get_pinyin(folder_name, '_')

    savepath = os.path.join(os.getcwd(), 'processed', folder_name + '_' + str(data_np.shape[0]) + '_'  + str(data_np.shape[1]) + '_'  + str(data_np.shape[2]) + '.nii.gz')
    # print(savepath)
    # savepath = savepath.encode('gbk').decode("utf-8")
    # savepath = pathlib.Path(savepath)
    # print(savepath)


    sitk.WriteImage(images, savepath )
    shutil.rmtree(path)


check_and_creat_dir('processed')

folder = 'extract'
all_folder = getfiles(folder)

for folder_name in all_folder:    
    print("processing" + folder_name)
    read_dcm(os.path.join(folder, folder_name), folder_name)


    # # 读取nii.gz并plot
    # filename = '2022-09-18 14.nii.gz'
    # img = sitk.ReadImage(filename)
    # data_np = sitk.GetArrayFromImage(img)
    # print(data_np.shape)
    # print(data_np.dtype)
    # tigre.plotImg(data_np.transpose(1,2,0))
