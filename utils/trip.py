import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython

### -----------------------------------------------
###
###  Utils function
###
### -----------------------------------------------

def dist_ortho(lon1, lat1, lon2, lat2):
    R = 6377726
    pi = np.pi
    a = np.sin((lat1 - lat2)/2*pi/180)**2
    b = np.cos(lat1*pi/180)*np.cos(lat2*pi/180)
    c = np.sin((lon1- lon2)/2* pi/180)**2

    dist = R * 2* np.arcsin( np.sqrt(a + b*c))
    return dist

def cap(lon1, lat1, lon2, lat2):
    pi = np.pi

    # to radians
    lat1 = lat1*pi/180
    lat2 = lat2*pi/180
    lon1 = lon1*pi/180
    lon2 = lon2*pi/180

    delta_lon = lon2-lon1

    a = np.cos(lat1) * np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(delta_lon)
    b = np.sin(delta_lon) * np.cos(lat2)

    cap = np.arctan2(b , a)
    cap = cap%(2*pi)

    return cap*180/pi

### -----------------------------------------------
###
###  Trip class to trajectory data
###
### -----------------------------------------------

class Trip:
    def __init__(self, df):

        self.df = df.set_index(np.arange(len(df)))
        self.threshold = None
        self.dist_matrix = None

    ### GENERAL STATISTICS ### -----------------------------------------------

    def get_duration(self):
        return max(self.df.datetime) - min(self.df.datetime)

    def get_distance(self):
        return sum(self.df.step_length)/1e3

    def standardize(self, vars):
        for var in vars:
            var_std = var + '_std'

            df_var = self.df[var].values
            self.df[var_std] = (df_var - np.nanmean(df_var))/np.nanstd(df_var)

    def standardize_minmax(self, vars):
        for var in vars:
            var_std = var + '_std_mm'

            df_var = self.df[var].values
            self.df[var_std] = (df_var - np.nanmin(df_var))/ (np.nanmax(df_var) - np.nanmin(df_var))

    ### DIVE DETECTION ### -----------------------------------------------

    def get_dive(self, threshold):
        self.threshold = threshold
        pressure = self.df.pressure.values
        bias = np.median(pressure)
        return 1*(pressure - bias > threshold)

    def add_dive(self, threshold):
        self.df['dive'] = self.get_dive(threshold)

    ### STEP-ANGLE PAIRS ### -----------------------------------------------

    def get_step(self):
        n = len(self.df)
        step = dist_ortho( self.df.lon.values[0:(n-1)], self.df.lat.values[0:(n-1)], self.df.lon.values[1:n], self.df.lat.values[1:n])
        return step

    def add_step(self):
        self.df['step'] = np.append(np.nan, self.get_step())

    def get_cap(self):
        n = len(self.df)
        c = cap( self.df.lon.values[0:(n-1)], self.df.lat.values[0:(n-1)], self.df.lon.values[1:n], self.df.lat.values[1:n])
        return c

    def add_cap(self):
        self.df['cap'] = np.append(np.nan, self.get_cap())

    def get_direction(self):
        direction = [d%360 - 360 if d%360 > 180 else d%360 for d in np.diff(self.get_cap())]
        return np.array(direction)

    def add_direction(self):
        a = np.empty(2)
        a.fill(np.nan)
        self.df['direction'] = np.append(a, self.get_direction())

    def plot(self, save = None):

        bias = np.median(self.df.pressure.values)

        plt.figure(figsize=(15, 3))

        plt.subplot(1, 3, 1)
        plt.plot(self.df.datetime.values, self.df.pressure.values)
        plt.plot(self.df.datetime.values, [self.threshold+bias for i in range(len(self.df))], color = 'orange')

        plt.subplot(1, 3, 2)
        plt.plot(self.df.lon.values, self.df.lat.values)
        plt.scatter(self.df.lon.values[self.df.pressure-bias > self.threshold], \
                    self.df.lat.values[self.df.pressure-bias > self.threshold], c = 'orange')

        plt.subplot(1, 3, 3)
        plt.scatter(self.df.step_direction, self.df.step_speed, alpha = 0.3)
        plt.scatter(self.df.step_direction.values[self.df.pressure-bias > self.threshold],
                    self.df.step_speed[self.df.pressure-bias > self.threshold], c = 'orange')

        if save is not None:
            plt.savefig(save)
            plt.close()

    ### AREA RESTRICTED RESEARCH ### -----------------------------------------------

    def compute_dist_matrix(self):
        # matrix of distance
        lon = np.vstack([self.df.lon.values for i in range(len(self.df))])
        lat = np.vstack([self.df.lat.values for i in range(len(self.df))])
        dd = dist_ortho(lon, lat, lon.T, lat.T)
        self.dist_matrix = dd

    def residence_time(self, radius):
        residence_matrix = self.dist_matrix < radius
        residence = [sum(residence_matrix[:,i]) for i in range(len(self.df))]

        return residence

    def first_time_passage(self, radius):
        first_passage = np.zeros(len(self.df))
        residence_matrix = self.dist_matrix < radius
        if np.sum(residence_matrix) > 0 :
            for i in range(len(self.df)):
                idx = np.where(residence_matrix[:,i])[0]
                delta = np.diff(idx)
                # get index start
                i_start = i
                while i_start in idx:
                    i_start -=1
                # get index end
                i_end = i
                while i_end in idx:
                    i_end +=1
                # get number of point
                first_passage[i] = i_end-i_start+1

        return first_passage

### -----------------------------------------------
###
###  Dataset class from pytorch to load and format data
###
### -----------------------------------------------
import torch
from torch.utils.data import Dataset

def standardize(var):
    var = np.array(var)
    return (var-np.mean(var))/np.std(var)

def standardize_minmax(var):
    var = np.array(var)
    return (var-np.min(var))/(np.max(var) - np.min(var))

def standardize_data(data):
    data = data.copy()
    data = data.dropna()
    data['lon_std'] = 0
    data['lat_std'] = 0
    data['step_speed_std'] = 0
    data['step_direction_cos'] = 0
    data['step_direction_sin'] = 0
    for trip in data.trip.unique():
        data.loc[data.trip == trip,'lon_std'] = standardize(data.loc[data.trip == trip,'lon'])
        data.loc[data.trip == trip, 'lat_std'] = standardize(data.loc[data.trip == trip,'lat'])
        data.loc[data.trip == trip,'step_speed_std'] = standardize_minmax(data.loc[data.trip == trip,'step_speed'])
        data.loc[data.trip == trip,'step_direction_cos'] = np.cos(data.loc[data.trip == trip,'step_direction']* np.pi/180)
        data.loc[data.trip == trip,'step_direction_sin'] = np.sin(data.loc[data.trip == trip,'step_direction']* np.pi/180)
    return data

def change_resolution(data, resolution):
    data_new = pd.DataFrame()
    for i in data.trip.unique():
        t = data[data.trip == i].copy()

        idx = [i%resolution == 0 for i in range(len(t))]

        traj = t.loc[idx, ('trip', 'datetime', 'lon', 'lat')]

        traj['dive'] = [np.max(t.dive[i:i+resolution]) for i in range(len(t)) if i%resolution==0]

        n = len(traj)
        step = dist_ortho( traj.lon.values[0:(n-1)], traj.lat.values[0:(n-1)], traj.lon.values[1:n], traj.lat.values[1:n])
        c = cap( traj.lon.values[0:(n-1)], traj.lat.values[0:(n-1)], traj.lon.values[1:n], traj.lat.values[1:n])
        direction = [d%360 - 360 if d%360 > 180 else d%360 for d in np.diff(c)]

        traj['step_speed'] = np.append(np.nan, step/resolution)
        traj['step_direction'] = np.append([np.nan, np.nan], direction)

        data_new = data_new.append(traj, ignore_index=True)
    return data_new


class TrajDataSet(Dataset):
    def __init__(self,  df, window, variable, transform=None):
        self.df = df.set_index(np.arange(len(df))) #reorder idx
        self.window = window
        self.var = variable
        self.start_idx = np.where([self.df.trip[i]==self.df.trip[i+self.window-1] for i in range(len(self.df)-self.window+1)])[0]
        self.transform = transform

    def __len__(self):
        return len(self.start_idx)

    def __getitem__(self, idx):

        i = self.start_idx[idx]

        # select variable of interest
        traj = self.df.loc[i:i+self.window-1, self.var]
        traj = np.array(traj).T

        # select coordinates
        coord = self.df.loc[i:i+self.window-1, ('lon', 'lat')]
        coord = np.array(coord).T
        lon = np.vstack([coord[0] for i in range(traj.shape[1])])
        lat = np.vstack([coord[1]  for i in range(traj.shape[1])])
        dd = dist_ortho(lon, lat, lon.T, lat.T)

        dive = self.df.loc[i:i+self.window-1, 'dive']
        dive = np.array(dive)

        sample = (traj, dd, dive)

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    def __init__(self, ratio, method='max'):
        self.ratio = ratio
        self.method = method

    def __call__(self, sample):
        traj, dd, dive = sample

        # change resolution
        if self.method == 'max':
            dive_new = [np.max(dive[i:i+self.ratio+1]) for i in range(len(dive)) if i%self.ratio==0]

        if self.method == 'mean':
            dive_new = [np.mean(dive[i:i+self.ratio+1]) for i in range(len(dive)) if i%self.ratio==0]

        return (traj, dd, dive_new)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        traj, dd, dive = sample
        traj, dd, dive = (torch.FloatTensor(traj), torch.FloatTensor(dd), torch.FloatTensor(dive))
        return (traj.unsqueeze(0), dd.unsqueeze(0),  dive.unsqueeze(0))
