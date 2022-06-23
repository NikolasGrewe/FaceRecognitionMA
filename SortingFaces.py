# -*- coding: utf-8 -*-
import shutil as st
import os

i = 0

for root, dirs, files in os.walk('pictures/unspecific/TrainingSet/Faces'):
    for filename in files:
        print(os.path.join(root, filename))
        st.move(os.path.join(root, filename), 'pictures/unspecific/TrainingSet/Faces/unsorted/face' + str(i) + '.jpg')
        i += 1