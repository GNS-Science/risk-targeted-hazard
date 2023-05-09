from .base import *
from .prepare_design_intensities import *

from toshi_hazard_store.query import get_hazard_curves
from nzshm_common.location.location import location_by_id, LOCATION_LISTS
from nzshm_common.location.location import LOCATIONS_BY_ID
from nzshm_common.location.code_location import CodedLocation
from nzshm_common.location import location
from nzshm_common.grids import RegionGrid


def save_hdf(hf_name, data):
    """
    Saves the data dictionary as an hdf5 file for later use.

    :param hf_name: name of the hdf5 file
    :param data: dictionary containing hazard curves and metadata for vs30, sites, intensity measures, and design intensities
    """
    with h5py.File(hf_name, 'w') as hf:

        # add metadata
        grp = hf.create_group('metadata')
        grp.attrs['vs30s'] = data['metadata']['vs30s']
        grp.attrs['quantiles'] = data['metadata']['quantiles']
        grp.attrs['acc_imtls'] = str(data['metadata']['acc_imtls'])
        grp.attrs['disp_imtls'] = str(data['metadata']['disp_imtls'])
        grp.attrs['sites'] = str(data['metadata']['sites'].to_dict())

        # add hazard curves
        grp = hf.create_group('hcurves')
        for dset_name in ['hcurves_stats']:
            dset = grp.create_dataset(dset_name, np.array(data['hcurves'][dset_name]).shape)
            dset[:] = np.array(data['hcurves'][dset_name])

        # add poe values
        if 'hazard_design' in data.keys():
            grp = hf.create_group('hazard_design')
            grp.attrs['hazard_rps'] = data['hazard_design']['hazard_rps']
            for intensity_type in ['acc', 'disp']:
                subgrp = grp.create_group(intensity_type)
                for dset_name in ['stats_im_hazard']:
                    dset = subgrp.create_dataset(dset_name,
                                                 np.array(data['hazard_design'][intensity_type][dset_name]).shape)
                    dset[:] = np.array(data['hazard_design'][intensity_type][dset_name])

        # add risk values
        if 'risk_design' in data.keys():
            grp = hf.create_group('risk_design')
            intensity_type = 'acc'
            subgrp = grp.create_group(intensity_type)
            subgrp.attrs['risk_assumptions'] = str(data['risk_design'][intensity_type]['risk_assumptions'])
            for grp_name in ['im_risk']:
                rlzgrp = subgrp.create_group(grp_name)
                for dset_name in ['im_risk', 'lambda_risk', 'fragility_risk', 'disagg_risk']:
                    dset = rlzgrp.create_dataset(dset_name, np.array(
                        data['risk_design'][intensity_type][grp_name][dset_name]).shape)
                    dset[:] = np.array(data['risk_design'][intensity_type][grp_name][dset_name])


def update_hdf_for_risk(hf_name, data):
    """
    Updates the hdf5 file to include the risk-informed design intensities.

    :param hf_name: name of the hdf5 file
    :param data: dictionary containing hazard curves and metadata for vs30, sites, intensity measures, and design intensities
    """
    with h5py.File(hf_name, 'r+') as hf:

        if 'risk_design' in hf.keys():
            del hf['risk_design']

        # add risk values
        if 'risk_design' in data.keys():
            grp = hf.create_group('risk_design')
            intensity_type = 'acc'
            subgrp = grp.create_group(intensity_type)
            subgrp.attrs['risk_assumptions'] = str(data['risk_design'][intensity_type]['risk_assumptions'])
            for grp_name in ['im_risk']:
                rlzgrp = subgrp.create_group(grp_name)
                for dset_name in ['im_risk', 'lambda_risk', 'fragility_risk', 'disagg_risk']:
                    dset = rlzgrp.create_dataset(dset_name, np.array(
                        data['risk_design'][intensity_type][grp_name][dset_name]).shape)
                    dset[:] = np.array(data['risk_design'][intensity_type][grp_name][dset_name])


def retrieve_site_design_intensities_stats(data_file, design_type, vs30, site, imt, metric, ls_key,
                                           intensity_type='acc'):
    if design_type == 'Hazard PoE':
        design_type = 'hazard_design'
    elif design_type == 'Risk Informed':
        design_type = 'risk_design'

    with h5py.File(data_file, 'r') as hf:
        vs30s = list(hf['metadata'].attrs['vs30s'])
        sites = pd.DataFrame(ast.literal_eval(hf['metadata'].attrs['sites']))
        imtls = ast.literal_eval(hf['metadata'].attrs['acc_imtls'])
        quantiles = list(hf['metadata'].attrs['quantiles'])

        if metric == 'mean':
            i_stat = 0
        else:
            i_stat = quantiles.index(metric) + 1

        idx_vs30 = vs30s.index(vs30)
        idx_site = list(sites.index).index(site)
        idx_imt = list(imtls.keys()).index(imt)

        if design_type == 'hazard_design':
            hazard_rps = hf[design_type].attrs['hazard_rps']
            idx_ls = hazard_rps.tolist().index(ls_key)
            design_im = hf[design_type][intensity_type]['stats_im_hazard'][idx_vs30, idx_site, idx_imt, idx_ls, i_stat]
        elif design_type == 'risk_design':
            if metric == 'mean':
                risk_assumptions = ast.literal_eval(hf['risk_design']['acc'].attrs['risk_assumptions'])
                idx_ls = list(risk_assumptions.keys()).index(ls_key)
                design_im = hf[design_type]['acc']['im_risk']['im_risk'][idx_vs30, idx_site, idx_imt, idx_ls]
                if intensity_type == 'disp':
                    design_im = acc_to_disp(design_im, period_from_imt(imt))
            else:
                raise NameError('Risk informed intensities are only provided for the mean hazard curve.')

        return design_im


def retrieve_site_hazard_curves_rlz(data_file, vs30, site, imt, i_rlz):
    with h5py.File(data_file, 'r') as hf:
        vs30s = list(hf['metadata'].attrs['vs30s'])
        sites = pd.DataFrame(ast.literal_eval(hf['metadata'].attrs['sites']))
        imtls = ast.literal_eval(hf['metadata'].attrs['acc_imtls'])

        idx_vs30 = vs30s.index(vs30)
        idx_site = list(sites.index).index(site)
        idx_imt = list(imtls.keys()).index(imt)

        return hf['hcurves']['hcurves_rlzs'][idx_vs30, idx_site, idx_imt, :, i_rlz]


def retrieve_site_hazard_curves_stats(data_file, vs30, site, imt, metric):
    with h5py.File(data_file, 'r') as hf:
        vs30s = list(hf['metadata'].attrs['vs30s'])
        sites = pd.DataFrame(ast.literal_eval(hf['metadata'].attrs['sites']))
        imtls = ast.literal_eval(hf['metadata'].attrs['acc_imtls'])
        quantiles = list(hf['metadata'].attrs['quantiles'])
        if metric == 'mean':
            i_stat = 0
        else:
            i_stat = quantiles.index(metric) + 1

        idx_vs30 = vs30s.index(vs30)
        idx_site = list(sites.index).index(site)
        idx_imt = list(imtls.keys()).index(imt)

        return hf['hcurves']['hcurves_stats'][idx_vs30, idx_site, idx_imt, :, i_stat]


def create_sites_df(named_sites=True, site_list=None, cropped_grid=False,
                    grid_limits=[-np.inf, np.inf, -np.inf, np.inf]):
    # create a dataframe with named sites
    if named_sites:
        id_list = LOCATION_LISTS['SRWG214']['locations']

        # if no list is passed, include all named sites
        if site_list is None:
            site_list = [location_by_id(loc_id)['name'] for loc_id in id_list]

        # collect the ids for the relevant sites
        id_list = [loc_id for loc_id in id_list if location_by_id(loc_id)['name'] in site_list]

        # create the df of named sites
        sites = pd.DataFrame(index=site_list, dtype='str')
        for loc_id in id_list:
            latlon = CodedLocation(location_by_id(loc_id)['latitude'], location_by_id(loc_id)['longitude'], 0.001).code
            lat, lon = latlon.split('~')
            sites.loc[location_by_id(loc_id)['name'], ['latlon', 'lat', 'lon']] = [latlon, lat, lon]

    # create a dataframe with latlon sites
    else:
        site_list = 'NZ_0_1_NB_1_1'
        resample = 0.1
        grid = RegionGrid[site_list]
        grid_locs = grid.load()

        # remove empty location
        l = grid_locs.index((-34.7, 172.7))
        grid_locs = grid_locs[0:l] + grid_locs[l + 1:]

        site_list = []
        for gloc in grid_locs:
            loc = CodedLocation(*gloc, resolution=0.001)
            loc = loc.resample(float(resample)) if resample else loc
            site_list.append(loc.resample(0.001).code)

        # create the df of gridded locations
        sites = pd.DataFrame(index=site_list, dtype='str')
        for latlon in site_list:
            lat, lon = latlon.split('~')
            sites.loc[latlon, ['latlon', 'lat', 'lon']] = [latlon, lat, lon]

        # remove sites based on latlon
        if cropped_grid:
            min_lat, max_lat, min_lon, max_lon = grid_limits
            sites['float_lat'] = [float(lat) for lat in sites['lat']]
            sites = sites[(sites['float_lat'] >= min_lat) & (sites['float_lat'] <= max_lat)].drop(['float_lat'], axis=1)
            sites['float_lon'] = [float(lon) for lon in sites['lon']]
            sites = sites[(sites['float_lon'] >= min_lon) & (sites['float_lon'] <= max_lon)].drop(['float_lon'], axis=1)

        sites.sort_values(['lat','lon'],inplace=True)

    return sites


def query_nshm_hcurves(hazard_id, sites, vs30_list, imt_list, imtl_list, agg_list):
    site_list = list(sites.index)

    hcurves = -1 * np.ones([len(vs30_list), len(site_list), len(imt_list), len(imtl_list), len(agg_list)])

    for i_site, site in enumerate(site_list):
        print(f'Site #{i_site} of {len(site_list)}.')
        latlon = sites.loc[site, 'latlon']
        for res in get_hazard_curves([latlon], vs30_list, [hazard_id], imt_list, agg_list):
            vs30 = res.vs30
            imt = res.imt
            agg = res.agg

            i_vs30 = vs30_list.index(vs30)
            i_imt = imt_list.index(imt)
            i_agg = agg_list.index(agg)

            hcurves[i_vs30, i_site, i_imt, :, i_agg] = [val.val for val in res.values]

    # # identify any missing data before throwing an error
    # if np.sum(hcurves < 0) != 0:
    #     vs30_idx, site_idx, imt_idx, imtl_idx, agg_idx = np.where(hcurves < 0)
    #
    #     print('\nMissing data from:')
    #     print(f'\t{[vs30_list[idx] for idx in np.unique(vs30_idx)]}')
    #     if len(np.unique(site_idx)) > 5:
    #         print(f'\t{[site_list[idx] for idx in np.unique(site_idx)[:5]]} and more...')
    #     else:
    #         print(f'\t{[site_list[idx] for idx in np.unique(site_idx)]}')
    #     print(f'\t{[imt_list[idx] for idx in np.unique(imt_idx)]}')
    #     print(f'\t{[agg_list[idx] for idx in np.unique(agg_idx)]}')
    #
    # assert np.sum(hcurves < 0) == 0, NameError('Resolve missing data.')

    return hcurves


def interpolate_hcurves(data, n_new_imtls=300):
    imtls = data['metadata']['acc_imtls']
    hcurves = data['hcurves']['hcurves_stats']

    imts = list(imtls.keys())
    imtl_list = imtls[imts[0]]
    min_imtl = imtl_list[0]
    max_imtl = imtl_list[-1]

    new_imtl_list = np.logspace(np.log10(min_imtl), np.log10(max_imtl), n_new_imtls)
    new_imtls = {}
    for imt in imts:
        new_imtls[imt] = list(new_imtl_list)

    n_vs30, n_sites, n_imts, n_imtls, n_stats = hcurves.shape
    new_hcurves = np.zeros([n_vs30, n_sites, n_imts, n_new_imtls, n_stats])

    for i_vs30 in range(n_vs30):
        for i_site in range(n_sites):
            for i_imt in range(n_imts):
                for i_stat in range(n_stats):
                    hcurve = hcurves[i_vs30, i_site, i_imt, :, i_stat]
                    new_hcurves[i_vs30, i_site, i_imt, :, i_stat] = np.exp(
                        np.interp(np.log(new_imtl_list), np.log(imtl_list), np.log(hcurve)))

    return new_imtls, new_hcurves


def convert_imtls_to_disp(acc_imtls):
    '''
    converts the intensity measure types and levels to spectral displacements
    '''
    disp_imtls = {}
    for acc_imt in acc_imtls.keys():
        period = period_from_imt(acc_imt)
        disp_imt = acc_imt.replace('A', 'D')

        disp_imtls[disp_imt] = acc_to_disp(np.array(acc_imtls[acc_imt]), period).tolist()

    return disp_imtls


def add_uniform_hazard_spectra(data,hazard_rps=np.array([25,50,100,250,500,1000,2500])):

    imtls = data['metadata']['acc_imtls']
    data['metadata']['disp_imtls'] = convert_imtls_to_disp(imtls)

    # get poe values
    print('Calculating PoE values.')
    data['hazard_design'] = {}
    data['hazard_design']['hazard_rps'] = hazard_rps.tolist()

    for intensity_type in ['acc','disp']:
        data['hazard_design'][intensity_type] = {}
        data['hazard_design'][intensity_type]['stats_im_hazard'] = calculate_hazard_design_intensities(data,hazard_rps,intensity_type)


def create_hdf5_for_nshm_query(hf_name, hazard_id, sites, vs30_list, imt_list, imtl_list, agg_list, hazard_rps,
                               interpolate=True):
    # create a temporary dictionary
    data = {}

    # prep metadata
    imtls = {}
    for imt in imt_list:
        imtls[imt] = imtl_list

    data['metadata'] = {}
    data['metadata']['quantiles'] = [float(q) for q in agg_list[1:]]
    data['metadata']['acc_imtls'] = imtls
    data['metadata']['sites'] = sites
    data['metadata']['vs30s'] = vs30_list

    # query the nshm
    data['hcurves'] = {}
    print('Querying hazard curves...')
    data['hcurves']['hcurves_stats'] = query_nshm_hcurves(hazard_id, sites, vs30_list, imt_list, imtl_list, agg_list)

    # interpolate the hazard curves in logspace
    if interpolate:
        print('Interpolating hazard curves.')
        new_imtls, new_hcurves = interpolate_hcurves(data, n_new_imtls=300)
        data['metadata']['acc_imtls'] = new_imtls
        data['hcurves']['hcurves_stats'] = new_hcurves

    # update the dictionary with the uniform hazard spectra
    add_uniform_hazard_spectra(data, hazard_rps)

    # save the dictionary to an hdf5 file
    save_hdf(hf_name, data)

    print(f'{hf_name} has been created.')
