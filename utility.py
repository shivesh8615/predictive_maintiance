import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class pre_process:
    def __init__(self):
        self.df_tele = pd.read_csv("PdM_telemetry.csv")
        self.df_sel = self.df_tele.loc[self.df_tele['machineID']==11].reset_index(drop=True)

    def create_feature(self.start, self.end):
  # create features from the selected machine
        self.pressure = self.df_sel.loc[self.start: self.end, 'pressure']
        self.timestamp = pd.to_datetime(self.df_sel.loc[self.start: self.end, 'datetime'])
        self.timestamp_hour = self.timestamp.map(lambda x: x.hour)
        self.timestamp_dow = self.timestamp.map(lambda x: x.dayofweek)
# apply one-hot encode for timestamp data
        self.timestamp_hour_onehot = pd.get_dummies(self.timestamp_hour).to_numpy()
# apply min-max scaler to numerical data
        scaler = MinMaxScaler()
        pressure = scaler.fit_transform(np.array(pressure).reshape(-1,1))
 # combine features into one
        feature = np.concatenate([pressure, self.timestamp_hour_onehot], axis=1)
        X = feature[:-1]
        y = np.array(feature[5:,0]).reshape(-1,1)
        return X, y, scaler

    def shape_sequence(arr, step, start):
        out = list()
        for i in range(start, arr.shape[0]):
            low_lim = i
            up_lim = low_lim + step
            out.append(arr[low_lim: up_lim])

            if up_lim == arr.shape[0]:
            # print(i)
              break

        out_seq = np.array(out)
        return out_seq

