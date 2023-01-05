import os
import json
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler
import numba as nb
from scipy import signal
from sklearn.pipeline import Pipeline, make_union, make_pipeline

class CreateDataset:
    '''
    Класс для создания датасета из json - файлов
    :param pathOfDumps: Папка с дампами json. По умолчанию '../walkDUMPS'
    '''

    def __init__(self, filenames, pathOfDumps=None):

        self.pathOfDumps = pathOfDumps
        self.filenames = [file[:-4] + '.json' for file in filenames]
        self.columns = None
        self.DataSet = None
        self.Target = None
        self.medianaOfSizes = None

    def __CreateListOfData(self):
        '''
        Функция вычисляет средний размер файлов в filenames по измерению 0
        :return: лист numpy --- весь датасет и средний размер временного ряда в файлах
        '''
        shapes = []
        count = 1
        allLists = []
        mydf = pd.DataFrame()
        for name in self.filenames[:10]:
            print('NAME',name)
            f = open(self.pathOfDumps + '/' + name, "r")
            try:
                data = json.loads(f.read())
                mydf = pd.DataFrame(data)
                print('!', mydf.shape)
                numMatrix = mydf.T.values.tolist()
                allLists.append(numMatrix)
                numFeachers = len(numMatrix[0])
                shapes.append(numFeachers)
                print('Add json ', count, ' of ', len(self.filenames), ' to Dataset')
            except ValueError:
                print('Do not add json ', count, ' of ', len(self.filenames), ' to Dataset')
            count += 1


        self.columns = mydf.columns.to_list()

        self.medianaOfSizes = int(np.median(shapes))

        AllArrays = np.array(allLists)
        AllArrays_columns = np.arange(AllArrays.shape[1])
        index = [0, 1, 2, 12, 13, 14]

        AllArrays_new_columns = np.delete(AllArrays_columns, index)
        self.columns = np.delete(self.columns, index)

        AllArrays_new = AllArrays[:, AllArrays_new_columns]
        print('fg')
        return AllArrays_new


    def __resamplingArray(self, AllArrays):
        '''
        Функция ресемплирует массивы под среднее значение длины
        :param allLists: лист из массивов различной длины
        :param mediana: размер под который их надо ресемплировать
        :return: рессемплированный датасет
        '''

        AllArrays = np.array(AllArrays)

        axisY = AllArrays.shape[0] # количество файлов
        axisZ = AllArrays.shape[1]  # количество фичей

        # AllArrays = np.array(allLists)

        NewArray = np.empty([axisY, self.medianaOfSizes, axisZ])
        for count in range(axisY):
            for col in range(axisZ):
                timeser = AllArrays[count][col]
                arr = TimeSeriesResampler(sz=self.medianaOfSizes).fit_transform(timeser)
                sarr = np.squeeze(arr)
                NewArray[count, : , col] = sarr
            G = NewArray[count, :, :]

            NewArray[count, :, :] = self.__numba_loops_fill(G) # Заменили Nan предыдущим значением в столюбце
        self.DataSet = NewArray #self.__roundColumns(NewArray) # Проходим по столбцам и округляем их, чтобы оставить только 0 или 1

    def __numba_loops_fill(self, arr):
        '''
        Заменили Nan предыдущим значением в столбце
        Не самое быстрое решение, но быстрее перевода в pandas
        :param arr: 2d массив
        :return: 2d массив без nan
        '''
        out = arr.copy()
        for row_idx in range(out.shape[0]):
            for col_idx in range(out.shape[1]):
                if np.isnan(out[row_idx, col_idx]):
                    out[row_idx, col_idx] = out[row_idx -1, col_idx ]
        return out


    def __trainTargetSplitlist(self, listArray):
        '''
        выделим таргет
        :param DataSet:
        :return: Target
        '''
        Target = []
        DataSet = []
        for file in listArray:
            Target.append(file[-1][0])
            DataSet.append(file[0:-1])

        TargetList = []

        if Target[0] is not None:    # когда мы читаем с файла у нас таргет не nan
            for trg in Target:
                ls = trg * np.ones(self.medianaOfSizes)
                TargetList.append(ls.tolist())
        else: # когда мы читаем отдельный файл таргет нан
            ls = np.empty(self.medianaOfSizes)
            ls[:] = np.nan
            TargetList = ls.tolist()

        self.Target = TargetList
        self.columns = self.columns[:-1]

        return DataSet


    # def __remove_features(self, array,columns, nameOfFeatures):
    #     '''
    #     По названию фичи удалет ее из датасекта
    #     :param array: 2d np array
    #     :param columns: список колонок в данном 2d массиве
    #     :param nameOfFeatures: list of str
    #     :return: np array
    #     '''
    #     for name in nameOfFeatures:
    #         if name in columns:
    #             index = self.columns.index(name)
    #             indexes_array = np.arange(array.shape[1]).tolist()
    #             indexes_array.pop(index)
    #             array = array[:, indexes_array]
    #             columns.remove(name)
    #         else:
    #             print('Не существует ', name)
    #             pass
    #     return array, columns

    def __appendFunc(self, array1, array2):
        '''
        Переписали Append под свои нужды.
        :param array1:
        :param array2:
        :return:
        '''
        if isinstance(array2, list):
            if isinstance(array2[0], str):
                array1 = list(array1)
                for el in array2:
                    array1.append(el)

            elif isinstance(array2[0], np.ndarray):
                for el in array2:
                    array1 = np.append(array1, el, axis=1)
        return array1

    def __getFreconcy(self, array):
        '''
        Получаем три высших частоты походки
        :param array:
        :param columns:
        :return:
        '''
        columns = self.columns.copy()
        array1 = list(array)

        for index in np.arange(len(self.columns)):
            name = columns[index]
            f, Pxx = signal.periodogram(array[:, index])
            top3_freq_indices = np.flip(np.argsort(Pxx), 0)[0:3]
            freqs = f[top3_freq_indices]
            power = Pxx[top3_freq_indices]
            freq1 = freqs[0] * np.ones(array.shape[0]).reshape(-1, 1)
            power1 = power[0] * np.ones(array.shape[0]).reshape(-1, 1)
            freq2 = freqs[1] * np.ones(array.shape[0]).reshape(-1, 1)
            power2 = power[1] * np.ones(array.shape[0]).reshape(-1, 1)
            freq3 = freqs[2] * np.ones(array.shape[0]).reshape(-1, 1)
            power3 = power[2] * np.ones(array.shape[0]).reshape(-1, 1)

            array = self.__appendFunc(array, [freq1, power1, freq2, power2, freq3, power3])
            columns = self.__appendFunc(columns, ['freq1_' + name, 'power1_' + name, 'freq2_' + name, 'power2_' + name, 'freq3_' + name, 'power3_' + name])

        return array, columns



    def __ProcessingAndFeatures(self, data):
        '''
        Добавдяем фичи и убираем ненужные.
        :param data: np array timeseries, по количеству фичей
        :return: датафрейм
        '''

        data, columns = self.__getFreconcy(data)
        # data, columns = self.__remove_features(data, columns, ['fps', 'time', 'frame', 'direction'])

        return data, columns

    def __GluingofDataSets(self):
        '''
        Рессемплируем весь датасет
        :return:
        '''

        GoodDataset = []
        for num in range(self.DataSet.shape[0]):
            sample, columns = self.__ProcessingAndFeatures(self.DataSet[num])
            GoodDataset.append(sample)
        static_features = columns[len(self.columns):]
        self.columns = columns
        GoodDataset_arr = np.array(GoodDataset)


        index_static_features = [self.columns.index(el) for el in static_features]
        dynamic_features = [name for name in self.columns if name not in static_features]
        index_dynamic_features = [self.columns.index(el) for el in dynamic_features]

        return GoodDataset_arr[:, :, index_dynamic_features ], GoodDataset_arr[:, 0, index_static_features ]



    def loadJson(self):
        '''
        загружаем json файлы, создаём numpy  датасет
        :return: Датасет
        '''


        flag = False
        for name in self.filenames:
            _, file_extension = os.path.splitext(name)
            if file_extension == '.json':
                flag = True
                break
        if flag == False:
            raise Exception('Директория ' + self.pathOfDumps +' не содержит ни ордного *.json файла')


        allArraysInDataSet = self.__CreateListOfData()  # загрузили всё в датасет

        SplittedArray = self.__trainTargetSplitlist(allArraysInDataSet) # Выделили Таргет и обновили имена столбцов
        self.__resamplingArray(SplittedArray) #Превели к одному размеру

        DataDynamic, DataStatic = self.__GluingofDataSets() # Добавили пару фичей и разбили на timeseries и статичные фичи для датасета



        return  DataDynamic, DataStatic, self.Target,  self.columns
