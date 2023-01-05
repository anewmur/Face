from CreateJsonFolder import CreateJsonFolder
from CreateDataset import CreateDataset
from PredictorOfState import PredictorOfState
from ModeLinference import ModeLinference
import pandas as pd
import numpy as np
import os


desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',15)



class Process:
    def __init__(self, MainPath, pathOfVideos, pathOfDumps, pathForModels):
        self.MainPath = MainPath
        self.pathOfVideos = pathOfVideos

        if self.MainPath is None:
            raise Exception('You must specify the path to the video files')

        _, self.file_extension = os.path.splitext(self.MainPath)

        if pathOfDumps is None and self.file_extension == '.csv': # работаем со списком файлов и определённым таргетом
            self.pathOfDumps = '../../faceDUMPS'
        elif pathOfDumps is None and self.file_extension == '.mp4': # работаем с деничным файлом
            basename = os.path.basename(self.MainPath)[:-4]
            self.pathOfDumps = '../../faceDUMPS_' + basename
        else:
            self.pathOfDumps = pathOfDumps
        if pathForModels is None:
            self.pathForModels = '../FaceModels'
        else:
            self.pathForModels = pathForModels

    def Learning(self, Practice, Facedetection = False):
        '''
        Функция, которая обучает модели по упражнению
        :param Practice: Упражнение
        :return:
        '''
        if self.file_extension != '.csv':
            raise Exception('It is not possible to train a model on one video!')

        JSF = CreateJsonFolder(path=self.MainPath, pathOfDumps=self.pathOfDumps, pathOfVideos=self.pathOfVideos,
                              Practice=Practice)


        _, filenames = JSF.GenerateJson(Facedetection=Facedetection, writeVideo=False)

        CD = CreateDataset(filenames=filenames, pathOfDumps=self.pathOfDumps)
        DataDynamic, DataStatic, Target, NamesOfColumns = CD.loadJson()

        print('PredictorOfState')
        PD = PredictorOfState(DataDynamic, DataStatic, Target, NamesOfColumns)
        clfArr, clf = PD.CascadLearning()

        MD = ModeLinference(pathForModels=self.pathForModels, NamesOfColumns=NamesOfColumns, DataDynamic=DataDynamic, DataStatic=DataStatic)
        MD.SaveModel(clfArr, clf, Practice)




if __name__ == "__main__":
    # ======================================================================================================
    # ОБУЧЕНИЕ МОДЕЛЕЙ


    # MainPath = '../../../../../Yandex.Disk.localized/_FEDOR_/data.csv'
    # pathOfVideos = '../../../../../Yandex.Disk.localized/_FEDOR_/VIDEO'

    # print(os.listdir('..\..\..\..\YandexDisk'))

    MainPath = '../../../../YandexDisk/Parkinson/VIDEOS/_FEDOR_/data.csv'
    pathOfVideos = '../../../../YandexDisk/Parkinson/VIDEOS/_FEDOR_/VIDEO'


    pathOfDumps = None # по умолчанию '../walkDUMPS'
    pathForModels = None  # по умолчанию '../Models'

    Models = Process(MainPath=MainPath, pathOfVideos=pathOfVideos, pathOfDumps=pathOfDumps, pathForModels=pathForModels)


    Practice = 'ЛИЦА'
    Models.Learning(Practice, Facedetection=True)
