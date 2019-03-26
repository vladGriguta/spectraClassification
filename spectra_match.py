import numpy as np
import pandas as pd
import time
import pickle
import re
import os
from astropy.io import fits

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
    fibre = int(re.findall(r"[\w']+",name)[3])
    
    match_found = False
    # Do a search in steps
    for i in range(len(matching)):
        if(matching['PLATE'].iloc[i] == plate):
            if(matching['MJD'].iloc[i] == mjd):
                if(matching['FIBERID'].iloc[i] == fibre):
                    match_found = True
                    class_ = matching['CLASS'].iloc[i]
                    subclass_ = matching['SUBCLASS'].iloc[i]
                    z_ = matching['Z'].iloc[i]
                    z_err = matching['Z_ERR'].iloc[i]
                    z_warn = matching['ZWARNING'].iloc[i]
                    best_obj = matching['BESTOBJID'].iloc[i]
                    #remove the matching
                    matching.drop(matching.data.index(i),inplace=True)
                    break
                    print(i)
    if(match_found):
        return matching,class_,subclass_,z_,z_err,z_warn,best_obj
    else:
        print(str(plate)+'-'+str(mjd)+str(fibre)+' not found.')
        raise Exception('No match found')

if __name__=='__main__':
    location_spectra = 'spectra_matched/'
    if not os.path.exists(location_spectra):
        os.makedirs(location_spectra)    

    f=fits.open('../matchData/specObj-dr14.fits', memmap=True)
    #had some memory issues since this is a huge file, i think memmap helped.
    # get the columns that we need
    columns = ['PLATE','MJD','FIBREID','CLASS','SUBCLASS','Z','Z_ERR','ZWARNING','BESTOBJID']
    matching = pd.DataFrame(columns=columns)
    matching['PLATE'] = f[1].data['PLATE']
    matching['MJD'] = f[1].data['MJD']
    matching['FIBERID'] = f[1].data['FIBERID']    
    matching['CLASS'] = f[1].data['CLASS']
    matching['SUBCLASS'] = f[1].data['SUBCLASS']
    matching['Z'] = f[1].data['Z']
    matching['Z_ERR'] = f[1].data['Z_ERR']
    matching['ZWARNING'] = f[1].data['ZWARNING']
    matching['BESTOBJID'] = f[1].data['BESTOBJID']
    
    print(len(matching))
    print(matching['CLASS'].iloc[0])
    print(f[1].data['CLASS'][0])

    start = time.time()
    import glob
    print('glob successfully imported.........................')
    filenames = glob.glob('../wgetThreading/spectraFull/*fits')
    print("ALL FILES OBTAINED. Number of them is "+str(len(filenames)))
    print ("obtaining filenames took "+ str(time.time() - start)+" seconds.")

    not_matched = []
    for i in range(len(filenames)):
        print(i)
        if(i%(len(filenames)/10000)==0):
            print("Progress is "+str(i/len(filenames))+" %")
        
        filename = filenames[i]
        # try to open file
        try:
            f = fits.open(filename)
        except:
            print('The element number '+str(i)+' could not be opened.')
            continue
        
        # now try to find match
        try:
            matching,class_,subclass_,z_,z_err,z_warn,best_obj = filename_match(filename,matching)
        except:
            print('No match was found for element '+str(i))
            not_matched.append(filename)
            continue

        # now add the matched objects to a dataframe
        columns=['flux','model','class','subclass','z','z_err','z_warn','best_obj']
        df_current = pd.DataFrame(columns=columns)
        df_current['flux'] = f[1].data['flux']
        df_current['model'] = f[1].data['model']
        df_current['class'] = class_
        df_current['subclass'] = subclass_
        df_current['z'] = z_
        df_current['z_err'] = z_err
        df_current['z_warn'] = z_warn
        df_current['best_obj'] = best_obj

        save_obj(df_current,name = (location_spectra+filename))

    save_obj(not_matched,name = (location_spectra+'notMatched'))
    
