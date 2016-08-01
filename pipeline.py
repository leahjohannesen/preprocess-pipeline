from preprocess import Preprocessing
import os
import pandas as pd
import numpy as np
import cPickle as pickle
import json

def get_score(json):

    model_list = []

    data = pd.DataFrame.from_dict(json)
    pre = Preprocessing()
    output = pre.process(data)

    predictions = np.array([])

    for idx, model_path in enumerate(model_list):
        with open(model_path) as pkl:
            model = pickle.load(pkl)
        x_arr = output[idx][0]
        y_pred = model.predict(x_arr)
        predictions = np.append(predictions, y_pred)

    pred = predictions.mean()

    return pred

if __name__ == '__main__':

    with open('example.json') as f:
        test_json = json.load(f)

    print get_score(test_json)
