from .base import *

def retrieve_data(file_id,named_sites=True):
    '''
    retrieves the relevant data and metadata from an oq .hdf5 file and stores it in a dictionary
    '''

    data = {}

    dstore = datastore.read(file_id)
    oqparam = vars(dstore['oqparam'])
    data['metadata'] = {}
    data['metadata']['quantiles'] = oqparam['quantiles']
    
    acc_imtls = oqparam['hazard_imtls']
    data['metadata']['acc_imtls'] = acc_imtls
    data['metadata']['disp_imtls'] = convert_imtls_to_disp(acc_imtls)
    
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

def convert_imtls_to_disp(acc_imtls):
    '''
    converts the intensity measure types and levels to spectral displacements
    '''
    disp_imtls = {}
    for acc_imt in acc_imtls.keys():
        period = period_from_imt(acc_imt)
        disp_imt = acc_imt.replace('A','D')

        disp_imtls[disp_imt] = acc_to_disp(np.array(acc_imtls[acc_imt]),period).tolist()
        
    return disp_imtls

def find_site_names(sites,dtol=0.1):
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