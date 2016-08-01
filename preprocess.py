import pandas as pd
import numpy as np
import helpers as helpers
import modelprocessors as modproc

class Preprocessing(object):

    def __init__(self):
        pass
        self.model_list = []

    def _y_strip(self, df, ycol):
        if ycol in df.columns:
            y_df = df[ycol].copy().to_frame()
            x_df = df.drop([ycol], axis=1)
            y_arr = y_df.values
        else:
            y_arr = None
        return df, y_arr

    def _global_pre(self, x_df):
        return x_df

    def _proc_0(self, x_df, y_arr):
        x_arr, y_arr = modproc.proc_0(x_df, y_arr)
        return x_arr, y_arr

    def process(self, df):
        model_list = [self._proc_0, self._proc_1, self._proc_2, self._proc_2, self._proc_2,
                      self._proc_2]

        x_df, y_arr = self._y_strip(df)
        x_df = self._global_pre(x_df)
        results = []

        for model in model_list:
            x_res, y_res = model(x_df, y_arr)
            results.append((x_res, y_res))
        return results

if __name__ == "__main__":

    train_path = 'data/train_new.json'

    train = pd.read_json(train_path)

    pre = Preprocessing()

    results = pre.process(train)
