import pickle
import os
import json
from CreateJsonFolder import CreateJsonFolder
import numpy as np

class ModeLinference():
    '''
    Кдасс для сохранения моделей и предсказания Паркинсона по сохраненённым моделям
    '''
    def __init__(self, pathForModels, NamesOfColumns, DataDynamic, DataStatic):
        self.pathForModels = pathForModels
        self.NamesOfColumns = NamesOfColumns
        self.DataDynamic = DataDynamic
        self.DataStatic = DataStatic

    def __renabePractice(self, Practice):
        '''
        Присваиваем псевдоним для сохранения и поиска моделей
        :param Practice: Упражнение
        :return:
        '''
        if Practice == 'ПОХОДКА':
            PracticeFile = 'Walk_model'
        elif Practice == 'ВСТАВАНИЕ СО СТУЛА':
            PracticeFile = 'StandUp_model'
        elif Practice == 'ПОДЪЕМ СТОПЫ':
            PracticeFile = 'FoolLift_model'
        elif Practice == 'ЛИЦА':
            PracticeFile = 'Face_model'
        else:
            raise Exception('Необходимо указать псевдоним для Practice в ModeLinference.__renabePractice')

        return PracticeFile

    def __MergeDatasets(self, ClassArr):
        '''
        Соединяем статические и динамические признаки
        :param ClassArr: Предсказания первого каскада
        :return: датасет для второго каскада
        '''
        Clall = np.array(ClassArr)

        Arr = []
        for col in range(self.DataDynamic.shape[0]):
            Arr1 = self.DataStatic[col, :]
            Arr2 = Clall[:, :, col].flatten()
            Arr.append(list(Arr1) + list(Arr2))

        Arr = np.array(Arr)

        return Arr

    def SaveModel(self, models1, model2, Practice):
        '''
        Функция сохраняет модели в папку self.pathForModels
        :param models1: Модели первого уровня каскадки
        :param model2: Модель второго уровня каскадки
        :param Practice: Управжнение
        :return:
        '''
        if not os.path.exists(os.path.join(os.getcwd(), self.pathForModels)):
            os.mkdir(self.pathForModels)

        PracticeFile = self.__renabePractice(Practice)

        if PracticeFile is not None:
            chet = 0
            for Dictmodels in models1:
                for model in Dictmodels:
                    md = Dictmodels[model]
                    with open(self.pathForModels + '/' + PracticeFile + '_' + str(self.NamesOfColumns[chet]) + '-' + str(model), 'wb') as f:
                        pickle.dump(md, f)
                chet +=1


            with open(self.pathForModels + '/' + PracticeFile + '_2', 'wb') as f:
                pickle.dump(model2, f)




    def predictMark(self, Practice):
        '''
        Предсказываем классы по обученным моделям
        :param Practice:
        :return:  Управжнение
        '''
        PracticeFile = self.__renabePractice(Practice)

        # Пройдёмся по папке с моделями и найдём все виды моделей, которые использовались на первом каскаде
        filenames = next(os.walk(self.pathForModels))[2]
        Allmodels = []
        for name in filenames:
            try:
                part1, part2 = name.split('-')
                Allmodels.append(part2)
            except ValueError:
                pass
        Allmodels = set(Allmodels)

        # Извлечём все модели
        NumFeaturesInDataDynamic = self.DataDynamic.shape[2]
        predArray = []

        # Последовательно проходимся по датасету и вызываем соответсвующие модели
        for feature in self.NamesOfColumns[:NumFeaturesInDataDynamic]:
            col = int(self.NamesOfColumns.index(feature))
            predArray1 = []
            for model in Allmodels:
                try:
                    with open(self.pathForModels + '/' + PracticeFile + '_' + str(feature) + '-' + str(model), 'rb') as f:
                        Learning_model = pickle.load(f)
                        Qdf = self.DataDynamic[:, :, col]
                        result = Learning_model.predict(Qdf)
                        predArray1.append(result.tolist())
                except ValueError:
                    pass
                predArray.append(predArray1)

        DataForSecondCascad = self.__MergeDatasets(predArray) # Массив для второй каскадки

        # загружаем модель второй каскадки и предсказываем по ней
        with open(self.pathForModels + '/' + PracticeFile + '_2', 'rb') as f:
            model2 = pickle.load(f)
            Mark = model2.predict(DataForSecondCascad)

        return Mark






