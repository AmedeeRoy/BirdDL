import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython


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

class Trip:
    def __init__(self, df):

        self.df = df.set_index(np.arange(len(df)))

    ### GENERAL STATISTICS ### -----------------------------------------------

    def get_duration(self):
        return max(self.df.datetime) - min(self.df.datetime)

    def get_distance(self):
        return sum(self.get_step())/1e3

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

    def plot(self):

        threshold = 1
        bias = np.median(df_trip.pressure.values)

        plt.figure(figsize=(15, 3))

        plt.subplot(1, 3, 1)
        plt.plot(self.df.datetime.values, self.df.pressure.values)
        plt.plot(self.df.datetime.values, [threshold+bias for i in range(len(self.df))], color = 'orange')

        plt.subplot(1, 3, 2)
        plt.plot(self.df.lon.values, self.df.lat.values)
        plt.scatter(self.df.lon.values[self.df.pressure-bias > threshold], \
                    self.df.lat.values[self.df.pressure-bias > threshold], c = 'orange')

        plt.subplot(1, 3, 3)
        plt.scatter(self.df.direction, self.df.step, alpha = 0.3)
        plt.scatter(self.df.direction.values[self.df.pressure-bias > threshold],
                    self.df.step.values[self.df.pressure-bias > threshold], c = 'orange')


    ### AREA RESTRICTED RESEARCH ### -----------------------------------------------

    def compute_dist_matrix(self, verbose=True):
        # matrix of distance
        self.dist_matrix = np.zeros((len(self.df), len(self.df)))

        for i in range(len(self.df)):
            lon1 = self.df.lon[[i for j in range(len(self.df))]]
            lat1 = self.df.lat[[i for j in range(len(self.df))]]
            lon2 = self.df.lon[[j for j in range(len(self.df))]]
            lat2 = self.df.lat[[j for j in range(len(self.df))]]

            self.dist_matrix[i, :] = dist_ortho(lon1.values, lat1.values, lon2.values, lat2.values)

            if verbose:
                IPython.display.clear_output(wait=True)
                print('Distance [{}/{}]'.format(i, len(self.df)))


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
