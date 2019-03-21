#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:59:30 2019

@author: vladgriguta

This script matches the fits files containing spectra with the photometric
objects 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from astropy.io import fits

# collect previous garbage
import gc
gc.collect()


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pd.DataFrame(pickle.load(f))
    
def closest_value(table, value,toleration):

    idx = np.argmin(np.abs(table - value))
    deviation = np.abs(table[idx] - value)
    if( deviation < toleration):
        return idx, deviation
    else:
        raise Exception('Above tolerations')



if __name__ == "__main__":
    toleration = 1e-5
    not_matched = []
    location_spectra = 'spectra_matched/'
    if not os.path.exists(location_spectra):
        os.makedirs(location_spectra)
    
    filename = '../moreData/test_query_table_4M'    
    data_table=load_obj(filename)
    
    # get the list of filenames
    import glob
    filenames = glob.glob('spectraFull/*.fits')
    
    for filename in filenames:
        f = fits.open(filename)
        # find closest value to curent RA
        try:
            idx, deviation = closest_value(data_table['#ra'],f[0].header['RA'],toleration)
            # Create a new dataframe to store the current spectra
            df_current = pd.DataFrame(columns=['flux','model'])
            df_current['flux'] = f[1].data['flux']
            df_current['model'] = f[1].data['model']
            df_current.objid = data_table['objid'].iloc[idx]
            df_current.classObj = data_table['class'].iloc[idx]
            df_current.subclassObj = data_table['subclass'].iloc[idx]
            df_current.z = data_table['z'].iloc[idx]
            df_current.header = f[0].header
            
            # save dataframe in location
            save_obj(df_current,name = (location_spectra+f[0].header['NAME']))
            
        except:
            """
            print('ERROR!! The difference between the matched RA and the current RA'+
                  ' was above the threshold of ' + str(deviation))
            """
            not_matched.append(f[0].header)
        
    save_obj(not_matched,name = (location_spectra+'notMatched'))
    
    