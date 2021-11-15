import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

### -----------------------------------------------
###
###  Utils function
###
### -----------------------------------------------

R = 6377726
pi = np.pi

def dist_ortho(lon1, lat1, lon2, lat2):
    a = np.sin((lat1 - lat2)/2*pi/180)**2
    b = np.cos(lat1*pi/180)*np.cos(lat2*pi/180)
    c = np.sin((lon1- lon2)/2* pi/180)**2
    dist = R * 2* np.arcsin( np.sqrt(a + b*c))
    return dist

def cap(lon1, lat1, lon2, lat2):
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
        traj['gaps'] = [np.mean(1*t.gaps[i:i+resolution]) for i in range(len(t)) if i%resolution==0]
        traj['dive'] = [np.max(t.dive[i:i+resolution]) for i in range(len(t)) if i%resolution==0]
        n = len(traj)
        step = dist_ortho( traj.lon.values[0:(n-1)], traj.lat.values[0:(n-1)], traj.lon.values[1:n], traj.lat.values[1:n])
        c = cap( traj.lon.values[0:(n-1)], traj.lat.values[0:(n-1)], traj.lon.values[1:n], traj.lat.values[1:n])
        direction = [d%360 - 360 if d%360 > 180 else d%360 for d in np.diff(c)]
        traj['step_speed'] = np.append(np.nan, step/resolution)
        traj['step_direction'] = np.append([np.nan, np.nan], direction)
        data_new = data_new.append(traj, ignore_index=True)
    return data_new

def format_data(data, resolution):
    data = change_resolution(data, resolution)
    data = standardize_data(data)
    return data
    
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
        dive = self.df.loc[i:i+self.window-1, 'dive']
        dive = np.array(dive)
        sample = (traj, dive)
        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    def __init__(self, ratio, method='max'):
        self.ratio = ratio
        self.method = method

    def __call__(self, sample):
        traj, dive = sample
        # change resolution
        if self.method == 'max':
            dive_new = [np.max(dive[i:i+self.ratio+1]) for i in range(len(dive)) if i%self.ratio==0]
        if self.method == 'mean':
            dive_new = [np.mean(dive[i:i+self.ratio+1]) for i in range(len(dive)) if i%self.ratio==0]
        return (traj, dive_new)

class Center(object):
    def __call__(self, sample):
        traj, dive = sample
        window = len(dive)
        dive_new = dive[int(window/2)]
        return (traj, np.array([dive_new]))

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        traj, dive = sample
        traj, dive = (torch.FloatTensor(traj), torch.FloatTensor(dive))
        return (traj, dive)

