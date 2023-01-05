import pandas as pd
import numpy as np

from pyts.classification import TimeSeriesForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_union, make_pipeline
from catboost import Pool, CatBoostClassifier
from sklearn.neural_network import MLPClassifier


class PredictorOfState:
    def __init__(self, DataDynamic, DataStatic, Target, NamesOfColumns):
        self.DataDynamic = DataDynamic
        self.DataStatic = DataStatic
        self.NamesOfColumns = NamesOfColumns
        self.Target = pd.DataFrame(Target).iloc[:, 0].tolist()

    def __StupidClassifiers(self, Qdf):
        '''
        Много классификаторов не бывает
        :param Qdf:
        :return:
        '''
        numUnic = len(set(self.Target))

        X_train, X_test, y_train, y_test = train_test_split(Qdf, self.Target, test_size=0.2, random_state=42)
        names = [
            "Pyts_TimeSeriesForest",
            "Nearest_Neighbors",
            "Linear_SVM",
            "RBF_SVM",
            "Decision_Tree",
            "Random_Forest",
            "AdaBoost",
            "Naive_Bayes",
            'MLPClassifier',
        ]

        classifiers = [
            TimeSeriesForest(n_estimators=200,
                             criterion='entropy',
                             max_features='sqrt',
                             verbose=0
                             ),
            KNeighborsClassifier(numUnic),
            SVC(kernel="linear", C=0.01),
            SVC(kernel="rbf", gamma=1, C=0.1),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=200, max_features=5),
            AdaBoostClassifier(),
            GaussianNB(),
            # MLPClassifier(max_iter=1500)
        ]

        predArray = []
        clfDict = {}
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            print(name, 'score is ', clf.score(X_test, y_test))
            predArray.append(clf.predict(Qdf))
            clfDict[name] = clf
        print('\n')

        return predArray, clfDict

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


    def __FirstLevelOfCascad(self):
        '''
        Первый уровень каскадки. Запускаем все классификаторы, ответы объединяем с частотными фичами
        :return Arr: Новый датссет
        '''
        ClassArr = []
        clfArray = []
        # print(self.DataDynamic.shape)

        for col in range(self.DataDynamic.shape[2]):
            oneClass, clfDict = self.__StupidClassifiers(self.DataDynamic[:, :, col])
            ClassArr.append(oneClass)
            clfArray.append(clfDict)

        return self.__MergeDatasets(ClassArr), clfArray

    def __SecondLevelOfCascad(self, Qdf):
        '''
        Второй уровень каскадки.
        :param Data: Датасет
        :return: Обученная модель
        '''
        X_train, X_test, y_train, y_test = train_test_split(Qdf, self.Target, test_size=0.3, random_state=42)

        # clf = RandomForestClassifier(max_depth=2, n_estimators=500, random_state=42)
        #
        # clf.fit(X_train, y_train)

        clf = CatBoostClassifier(iterations=50,
                                 learning_rate=0.1,
                                 depth=15,
                                 loss_function='MultiClass',
                                 verbose=0)

        clf.fit(X_train, y_train)

        print('Total score is ', clf.score(X_test, y_test))
        print('f1_score is ', f1_score(y_test, clf.predict(X_test), average='weighted'))
        print('precision_score is ', precision_score(y_test, clf.predict(X_test), average='weighted'))
        print('recall_score is ', recall_score(y_test, clf.predict(X_test), average='weighted'))
        return clf

    def CascadLearning(self):
        Arr, clfArr = self.__FirstLevelOfCascad()
        clf = self.__SecondLevelOfCascad(Arr)

        return clfArr, clf
