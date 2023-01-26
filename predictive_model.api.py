# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:12:58 2023

@author: SUNOJ CH
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('D:/cancer/trained_model.sav','rb'))

input_data = (1,69,1,2,2,1,1,2,1,2,2,2,2,2,2)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diagnosed with lung cancer')
else:
  print('The person is diagnosed with lung cancer')