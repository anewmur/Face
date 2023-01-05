
from Detector import Detector
import os
import pandas as pd
import numpy as np
import cv2


class CreateJsonFolder:
    '''
        Класс для создания json файлов с фичами из видеофайлов.
        :path: str ссыдка на документ *.cv2 или ссылку на *.mp4
        :pathOfDumps: str ссылка на документ куда складывать дампы, если None, то pathOfDumps = '../walkDUMPS'
        :pathOfVideos: str ссылка на папку с видеофайлами. Необходим только если path содержит ссылку на *.csv,
                       если  path содержит ссылку на *.mp4, то pathOfVideos приравнивается к папке с *.mp4
        :partOfBodyForTests: str часть тела для обработки медицинских тестов (пока реализовано только для "ПОХОДКА")
        :return: Список из названий файлов *.mp4
    '''

    def __init__(self, path, pathOfDumps, pathOfVideos=None, Practice=None):
        self.path = path
        self.target = None
        self.pathOfDumps = pathOfDumps
        self.pathOfVideos = pathOfVideos
        self.Practice = Practice
        self.dictTarget = {}
        self.filenames = None

    def __ArrayOfNames(self):
        '''
        Обрабатываем ссылку и получаем список видео файлов с которыми будем дальше работать
        :return: собственно этот массив
        '''
        _, file_extension = os.path.splitext(self.path)
        basename = os.path.basename(self.path)
        if file_extension == '.csv':
            if self.pathOfVideos is None:
                raise Exception('Необходимо указать ссылку на папку с видео в pathOfVideos ')
            try:
                df = pd.read_csv(self.path,
                                 encoding='cp1251',
                                 sep=',')
                df = df[df['УПРАЖНЕНИЕ'].str.contains(self.Practice)]
                outArray = df['ВИДЕО'].tolist()
                df = self.__codingTarget(df, 'ОЦЕНКА')
                self.target = df['ОЦЕНКА']
                self.target = self.target.to_list()
            except KeyError:
                print('Неправильный формат файла')

        elif file_extension == '.mp4':
            outArray = [basename]
            self.pathOfVideos = os.path.dirname(os.path.abspath(self.path))
        else:
            raise Exception('Что за хрень на входе?')

        return outArray

    def __codingTarget(self, df, column):
        '''
        Функция превращает категариальные переменные в датафрейме в численные. Почему то factorize не работает
        :param df: датафрейм
        :param column: название столбца с категориальными переменными
        :return: датафрейм без категориальных переменных
        '''
        unicueTarget = df[column].unique()
        numericTarget = np.arange(len(unicueTarget))
        for num in numericTarget:
            self.dictTarget[num] = unicueTarget[num]
        df[column].replace(unicueTarget, numericTarget, inplace=True)

        return df

    def __videoprocessing(self, file_name, Trg, writeVideo, Facedetection):
        if Facedetection:
            print(file_name)
            cap = cv2.VideoCapture(self.pathOfVideos + '/' + file_name)

            recog = Detector(self.pathOfVideos + '/' + file_name, Trg)
            recog.run(self.pathOfDumps, writeVideo)

    def GenerateJson(self, writeVideo, Facedetection):
        '''
        Создаём json файлы с извлечёнными данными
        :writeVideo': если True, то будет записан файл с демонстрацией фичей на видео в папку  '../procVideos'
        :return: json фйлоы с извлёчёнными признаками
        '''
        self.filenames  = self.__ArrayOfNames()
        count = 1
        for file_name in self.filenames :
            print('video ', count, ' of ', len(self.filenames ))

            Trg = None
            if self.target is not None:
                Trg = self.target[count - 1]


            if os.path.exists(self.pathOfDumps + '/' + file_name[:-4] + '.json'): # Проверяем Существует ли json
                # если да, - ничего не делаем
                print('Файл ' + file_name[:-4] + '.json' + ' в папке ' + self.pathOfDumps + ' уже существует')
            else:
                # если нет, печатеем это
                if not os.path.exists(self.pathOfVideos + '/' + file_name):
                    print('Видео ' + file_name + ' в папке ' + self.pathOfVideos + ' не найдено')
                else:
                    # и запускаем обработку видео
                    print('Видео ' + file_name + ' в папке ' + self.pathOfVideos + ' найдено')
                    self.__videoprocessing(file_name, Trg, writeVideo, Facedetection)

            count += 1

        return self.dictTarget, self.filenames
