from .base import *

from toshi_hazard_store.query_v3 import get_hazard_curves
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