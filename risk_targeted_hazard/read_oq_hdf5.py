from openquake.commonlib import datastore

import h5py
import pandas as pd
import numpy as np
import json
import glob
import re
from pathlib import Path


def retrieve_data(file_id,named_sites=True):
    '''
    retrieves the relevant data and metadata from an oq .hdf5 file and stores it in a dictionary
    '''

    data = {}

    dstore = datastore.read(file_id)
    oqparam = vars(dstore['oqparam'])
    data['metadata'] = {}
    data['metadata']['imtls'] = oqparam['hazard_imtls']
  
    if named_sites:
        data['metadata']['sites'] = find_site_names(dstore.read_df('sitecol')).to_dict()
    else:
        data['metadata']['sites'] = dstore.read_df('sitecol').to_dict()

    dstore.close()

    with h5py.File(file_id, 'r') as hf:
        data['metadata']['rlz_weights'] = hf['weights'][:].tolist()
        
        data['hcurves'] = {}
        data['hcurves']['hcurves_rlzs'] = np.moveaxis(hf['hcurves-rlzs'][:], 1, 3).tolist()
        data['hcurves']['hcurves_stats'] = np.moveaxis(hf['hcurves-stats'][:], 1, 3).tolist()

    return data


def find_site_names(sites,dtol=0.001):
    '''
    sets site names as the index for the sites dataframe
    '''
    
    ## this should be coming from the repo but I can't get the ssh set up yet
    with open('data/locations.json','r') as f:
        location_codes_list = json.load(f)
        location_codes = {}
        for loc in location_codes_list:
            location_codes[loc['name']] = {'id':loc['id'],'latitude':loc['latitude'],'longitude':loc['longitude']}
        location_codes = pd.DataFrame(location_codes).transpose()
    
    for i in sites.index:
        lat_idx = (location_codes['latitude'] >= sites.loc[i,'lat']-dtol) & (location_codes['latitude'] <= sites.loc[i,'lat']+dtol)
        lon_idx = (location_codes['longitude'] >= sites.loc[i,'lon']-dtol) & (location_codes['longitude'] <= sites.loc[i,'lon']+dtol)
        sites.loc[i,'name'] = location_codes[lat_idx & lon_idx].index
        
    return sites.set_index('name')