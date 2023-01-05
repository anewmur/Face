
import pandas as pd
import cv2
import json
from os.path import exists
import RECOG


df = pd.read_csv('../data.csv',
                 encoding = 'cp1251',
                 sep = ',')


df = df[df['УПРАЖНЕНИЕ'].str.contains('ЛИЦА')]

print(df.shape)

videos = df['ВИДЕО']

print(videos)


for test_video in videos:
    name = test_video[:-4] + '.json'
    file_exists = exists('../DUMPS/' + name)

    # if not file_exists:
    print(name)
    cap = cv2.VideoCapture('../VIDEO/' + test_video)

    det = RECOG.Detector(cap, '../DUMPS/' + name, is_file = True)
    det.run()
