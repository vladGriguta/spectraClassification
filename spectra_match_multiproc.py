import numpy as np
import pandas as pd
import time
import pickle
import re
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import multiprocessing
import itertools


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pd.DataFrame(pickle.load(f))

def filename_match(filename,matching):
    name = filename.split('/')[len(filename.split('/'))-1]
    plate = int(re.findall(r"[\w']+",name)[1])
    mjd = int(re.findall(r"[\w']+",name)[2])
    fiber = int(re.findall(r"[\w']+",name)[3])
    
    match_found = False
    matched_plate = matching[matching['PLATE'] == plate]
    #print('Plate...... '+str(len(matched_plate)))
    matched_plate_mjd = matched_plate[matched_plate['MJD'] == mjd]
    #print('plate-mjd.. '+str(len(matched_plate_mjd)))
    matched_all = matched_plate_mjd[matched_plate_mjd['FIBERID'] == fiber]
    print('all........ '+str(len(matched_all)))
    if(len(matched_all)>0):
        match_found = True
        class_ = matched_all['CLASS'].iloc[0]                                         
        subclass_ = matched_all['SUBCLASS'].iloc[0]                                            
        z_ = matched_all['Z'].iloc[0]                                                          
        z_err = matched_all['Z_ERR'].iloc[0]                                                   
        z_warn = matched_all['ZWARNING'].iloc[0]                                               
        best_obj = matched_all['BESTOBJID'].iloc[0]
        instrument = matched_all['INSTRUMENT'].iloc[0]
        #remove the matching cannot be done if multiproc

    if(match_found):
        return class_,subclass_,z_,z_err,z_warn,best_obj,instrument,name.split('.')[0]
    else:
        print(str(plate)+'-'+str(mjd)+str(fiber)+' not found.')
        raise Exception('No match found')

def save_matched_spectra(varyingData,constantData):
    filename = varyingData
    [matching,location_spectra] = constantData
    # try to open file
    try:
        f = fits.open(filename)
    except:
        print('The element number '+filename+' could not be opened.')
        return filename
    
    # now try to find match
    try:
        class_,subclass_,z_,z_err,z_warn,best_obj,instrument,name = filename_match(filename,matching)
    except:
        print('No match was found for element '+filename)
        return filename

    # now add the matched objects to a dataframe
    columns=['flux','model','loglam','information']
    df_current = pd.DataFrame(columns=columns)
    df_current['flux'] = f[1].data['flux']
    df_current['model'] = f[1].data['model']
    df_current['loglam'] = f[1].data['loglam']
    df_current['information'].iloc[0] = class_
    df_current['information'].iloc[1] = subclass_
    df_current['information'].iloc[2] = z_
    df_current['information'].iloc[3] = z_err
    df_current['information'].iloc[4] = z_warn
    df_current['information'].iloc[5] = best_obj
    df_current['information'].iloc[6] = instrument
    save_obj(df_current,name = (location_spectra+name))
    return 'Successful'
    
    

if __name__=='__main__':
    location_spectra = 'spectra_matched_new/'
    if not os.path.exists(location_spectra):
        os.makedirs(location_spectra)    

    f=fits.open('../matchData/specObj-dr14.fits', memmap=True)
    #had some memory issues since this is a huge file, i think memmap helped.
    # get the columns that we need
    columns = ['PLATE','MJD','FIBERID','CLASS','SUBCLASS','Z','Z_ERR','ZWARNING','BESTOBJID','INSTRUMENT']
    matching = pd.DataFrame(columns=columns)
    from astropy.table import Table
    t = Table.read(f[1])
    matching = t[columns].to_pandas()
    matching = pd.DataFrame(matching)

    start = time.time()
    import glob
    print('glob successfully imported.........................')
    filenames = glob.glob('../wgetThreading/spectraFull/*fits') # '' for the lofar machine
    print("ALL FILES OBTAINED. Number of them is "+str(len(filenames)))
    print ("obtaining filenames took "+ str(time.time() - start)+" seconds.")
    
    freeProc = 2
    n_proc=multiprocessing.cpu_count()-freeProc
    
    varyingData = filenames
    constantData = [matching,location_spectra]
    with multiprocessing.Pool(processes=n_proc) as pool:
        result_list=pool.starmap(save_matched_spectra, zip(varyingData, itertools.repeat(constantData)))
        pool.close()
    
    save_obj(result_list,name = (location_spectra+'notMatched'))
    
