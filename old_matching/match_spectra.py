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
import os
import pickle
from astropy.io import fits
import time

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

def closest_value(table, ra, dec, tolerance):
    
    current_distance = distance_on_sphere(np.array(data_table['#ra']-ra),
                                        np.array(data_table['dec']-dec))
    idx_min = np.argmin(current_distance)
    deviation_distance = current_distance[idx_min]
    print(deviation_distance)
    if( deviation_distance < tolerance):
        print('ra dif is '+str(np.abs(table['#ra'].iloc[idx]-ra)))
        print('dec dif is '+str(np.abs(table['dec'].iloc[idx]-dec)))
        if( (np.abs(table['#ra'].iloc[idx]-ra)<1) & (np.abs(table['dec'].iloc[idx]-dec) < 1)):
            print(deviation_distance)
            return idx, deviation_distance
    else:
        #print("Object not identified.....................")
        raise Exception('Above tolerations')



if __name__ == "__main__":
    tolerance = 1e-5
    not_matched = []
    location_spectra = 'spectra_matched/'
    if not os.path.exists(location_spectra):
        os.makedirs(location_spectra)
    
    filename = '../moreData/test_query_table_4M'    
    data_table=load_obj(filename)
    
    start = time.time()
    # get the list of filenames
    print("Atempting at getting the filenames of the spectra....")
    import glob
    print("glob imported successfuly............................")
    filenames = glob.glob('spectraFull/*.fits')
    print("ALL FILES OBTAINED. Number of them is "+str(len(filenames)))
    print ("obtaining filenames took "+ str(time.time() - start)+" seconds.")
    

    
    i = 0
    for filename in filenames:
        i += 1
        if(i%(len(filenames)/100)==0):
            print("Progress is "+str(i/len(filenames))+" %")
        f = fits.open(filename)
        # find closest value to curent RA
        try:
            #print("Try new spectra")
            idx, deviation = closest_value( table = data_table[['#ra','dec']], ra = f[0].header['RA'],
                                                                dec = f[0].header['DEC'], tolerance = tolerance)
            
            #print("Value found..................................")
            # Create a new dataframe to store the current spectra
            columns=['flux','model','class','subclass','z']
            df_current = pd.DataFrame(columns=columns)
            #print("DF created...................................")
            df_current['flux'] = f[1].data['flux']
            df_current['model'] = f[1].data['model']
            df_current['class'] = data_table['class'].iloc[idx]
            df_current['subclass'] = data_table['subclass'].iloc[idx]
            df_current['z'] = data_table['z'].iloc[idx]
            #print("spectra added................................")
            df_current.header = f[0].header
            
            # save dataframe in location
            save_obj(df_current,name = (location_spectra+f[0].header['NAME']))
            #del df_current
            #filenames_saved = glob.glob(location_spectra)
            #print(str(len(filenames_saved))+' files were saved.....................')
        except:
            """
            print('ERROR!! The difference between the matched RA and the current RA'+
                  ' was above the threshold of ' + str(deviation))
            """
            not_matched.append(f[0].header)
        
    save_obj(not_matched,name = (location_spectra+'notMatched'))
    
    
    
    