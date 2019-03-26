"""
@author: vladgriguta
This script checks the distance to the closest objects in the pkl files
"""
import numpy as np
import pandas as pd
import os
import pickle
from astropy.io import fits
import datetime

# collect previous garbage
import gc
gc.collect()


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pd.DataFrame(pickle.load(f))

def distance_on_sphere(ra,dec):
    return np.sqrt(np.square(np.multiply(ra,np.cos(dec))) + np.square(dec))


def check_distance(table):
    array_min_dist = []
    for i in range(1000):
        print(i)
        if( i % (len(table)/100) == 0):
            print('progress is ' + str(round(100*i/len(table))))
        ra = table['#ra'].iloc[i]
        dec = table['dec'].iloc[i]
        #time1 = datetime.datetime.now()
        current_distance = distance_on_sphere(np.array(table['#ra']-ra), np.array(table['dec']-dec))
        #print('operation took '+str(datetime.datetime.now()-time1))
        array_min_dist.append(np.min(current_distance[current_distance>0]))
    return array_min_dist


if __name__ == '__main__':
    
    filename = '../moreData/test_query_table_4M'    
    data_table=load_obj(filename)
    reduced_table = data_table[['#ra','dec']]
    
    array_min_dist = check_distance(reduced_table)
    print('mean is: ' + str(np.mean(array_min_dist)))
    print('std is: ' + str(np.std(array_min_dist)))