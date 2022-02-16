import json
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    # retrieve model by name
    path = Model.get_model_path('automl-model')
    model = joblib.load(path)

def run(input_data):
    data = json.loads(input_data)
    data_pred = pd.DataFrame(data, index=[0])
    # make prediction 
    y = model.predict(data_pred)
    return json.dumps({"classification": int(y[0])})

