from .base import *

def calculate_hazard_design_intensities(data,hazard_rps,intensity_type='acc'):
    '''
    calculate design intensities based on an annual probability of exceedance (APoE)

    :param data: dictionary containing hazard curves and metadata for sites, intensity measures, and rlz weights
    :param hazard_rps: np array containing the desired return periods (1 / APoE)

    :return: np arrays for all intensities from the hazard curve realizations and stats (mean and quantiles)
    '''
    
    imtls = data['metadata'][f'{intensity_type}_imtls']    
    rlz_weights = np.array(data['metadata']['rlz_weights'])
    hcurves_rlzs = np.array(data['hcurves']['hcurves_rlzs'])
    hcurves_stats   = np.array(data['hcurves']['hcurves_stats'])
    
    [n_sites,n_imts,n_imtls,n_rlz] = hcurves_rlzs.shape
    [_,_,_,n_stats] = hcurves_stats.shape
    
    n_rps = len(hazard_rps)
    
    im_hazard = np.zeros([n_sites,n_imts,n_rps,n_rlz,2])
    stats_im_hazard = np.zeros([n_sites,n_imts,n_rps,n_stats])
    
    for i_site in range(n_sites):
        for i_imt,imt in enumerate(imtls.keys()):
            for i_rlz in range(n_rlz):
                # logspace interpolation at the APoE for each return period
                im_hazard[i_site,i_imt,:,i_rlz,0] = np.exp(np.interp(np.log(1/hazard_rps), np.log(np.flip(hcurves_rlzs[i_site,i_imt,:,i_rlz])), np.log(np.flip(imtls[imt]))))

            # record the position of the realizations in the cdf of the full distribution
            for i_rp,rp in enumerate(hazard_rps):
                # order the metric to find the quantiles
                cdf_idx = np.argsort(np.squeeze(im_hazard[i_site,i_imt,i_rp,:,0]))
                im_idx = np.argsort(cdf_idx)
                cdf = np.cumsum(rlz_weights[cdf_idx])
                im_hazard[i_site,i_imt,i_rp,:,1] = cdf[im_idx]

            # loop over the median and any quantiles
            for i_stat in range(n_stats):
                stats_im_hazard[i_site,i_imt,:,i_stat] = np.exp(np.interp(np.log(1/hazard_rps), np.log(np.flip(hcurves_stats[i_site,i_imt,:,i_stat])), np.log(np.flip(imtls[imt]))))
                
    return im_hazard, stats_im_hazard


def calculate_risk_design_intensities(data,risk_assumptions,imtl_list):
    '''
    calculate design intensities based on a risk target and fragility assumptions

    :param data: dictionary containing hazard curves and metadata for sites, intensity measures, and rlz weights
    :param risk_target_assumptions: dictionary with keys for combinations of assumptions
    :param imtl_list: a list of intensity measures to include (must be included in the available imtls)

    :return: np arrays for all intensities from the hazard curve realizations and stats (mean and quantiles)
    '''

    intensity_type = 'acc'
    imtls = data['metadata'][f'{intensity_type}_imtls']
    rlz_weights = np.array(data['metadata']['rlz_weights'])
    hcurves_rlzs = np.array(data['hcurves']['hcurves_rlzs'])
    hcurves_stats = np.array(data['hcurves']['hcurves_stats'])

    [n_sites, n_imts, n_imtls, n_rlz] = hcurves_rlzs.shape
    [_, _, _, n_stats] = hcurves_stats.shape
    
    n_risk_assumptions = len(risk_assumptions.keys())
    
    im_risk = np.zeros([n_sites,n_imts,n_risk_assumptions,n_rlz,2])
    lambda_risk = np.zeros([n_sites,n_imts,n_risk_assumptions,n_rlz])
    fragility_risk = np.zeros_like(lambda_risk)
    
    stats_im_risk = np.zeros([n_sites,n_imts,n_risk_assumptions,n_stats])
    stats_lambda_risk = np.zeros_like(stats_im_risk)
    stats_fragility_risk = np.zeros_like(stats_im_risk)
    
    
    for imt in imtl_list:
        print(f'Processing {imt}.')
        i_imt = list(imtls.keys()).index(imt)
        for i_site in range(n_sites):
            # loop over the risk target assumption dictionaries
            for i_rt,rt in enumerate(risk_assumptions.keys()):
                risk_target = risk_assumptions[rt]['risk_target']
                beta = risk_assumptions[rt]['beta']
                conditional_prob = risk_assumptions[rt]['design_point']

                # optimize the design intensity for the risk target for each realization
                for i_rlz in range(n_rlz):
                    [im_r,median] = find_uniform_risk_intensity(hcurves_rlzs[i_site,i_imt,:,i_rlz], imtls[imt], beta, risk_target, conditional_prob)
                    im_risk[i_site,i_imt,i_rt,i_rlz,0] = im_r
                    lambda_risk[i_site,i_imt,i_rt,i_rlz] = np.interp(im_r, imtls[imt], hcurves_rlzs[i_site,i_imt,:,i_rlz])
                    fragility_risk[i_site,i_imt,i_rt,i_rlz] = median

                # record the position of the realizations in the cdf of the full distribution
                # order the metric to find the quantiles
                cdf_idx = np.argsort(np.squeeze(im_risk[i_site,i_imt,i_rt,:,0]))
                im_idx = np.argsort(cdf_idx)
                cdf = np.cumsum(rlz_weights[cdf_idx])
                im_risk[i_site,i_imt,i_rt,:,1] = cdf[im_idx]

                # loop over the median and any quantiles
                for i_stat in range(n_stats):
                    [im_r,median] = find_uniform_risk_intensity(hcurves_stats[i_site,i_imt,:,i_stat], imtls[imt], beta, risk_target, conditional_prob)
                    stats_im_risk[i_site,i_imt,i_rt,i_stat] = im_r
                    stats_lambda_risk[i_site,i_imt,i_rt,i_stat] = np.interp(im_r, imtls[imt], hcurves_stats[i_site,i_imt,:,i_stat])
                    stats_fragility_risk[i_site,i_imt,i_rt,i_stat] = median

    # store results as a dictionary
    im_risk = {'im_risk':im_risk.tolist(),'lambda_risk':lambda_risk.tolist(),'fragility_risk':fragility_risk.tolist()}
    stats_im_risk = {'stats_im_risk':stats_im_risk.tolist(),'stats_lambda_risk':stats_lambda_risk.tolist(),'stats_fragility_risk':stats_fragility_risk.tolist()}
    return im_risk, stats_im_risk



def risk_convolution_error(median, hcurve, imtl, beta, target_risk):
    '''
    error function for optimization

    :param median: median of the fragility function
    :param hcurve: hazard curve
    :param imtl:   intensity measure levels
    :param beta:   log std for the fragility function
    :param target_risk:  risk value to target

    :return: error from risk target
    '''

    pdf_limitstate_im = stats.lognorm(beta, scale=median).pdf(imtl)
    disaggregation = pdf_limitstate_im * hcurve
    risk = np.trapz(disaggregation, x=imtl)

    return np.abs(target_risk - risk)


def find_uniform_risk_intensity(hcurve, imtl, beta, target_risk, design_point):
    '''
    optimization to find the fragility and associated design intensity

    :param hcurve: hazard curve
    :param imtl:   intensity measure levels
    :param beta:   log std for the fragility function
    :param target_risk:   risk value to target
    :param design_point:  design point for selecting the design intensity

    :return: design intensity and median of fragility
    '''

    x0 = 0.5
    median = minimize(risk_convolution_error, x0, args=(hcurve, imtl, beta, target_risk), method='Nelder-Mead').x[0]
    im_r = stats.lognorm(beta, scale=median).ppf(design_point)

    return im_r, median


def risk_convolution(hcurve, imtl, median, beta):
    '''
    calculates the total annual risk and the underlying disaggregation curve

    :param hcurve: hazard curve
    :param imtl:   intensity measure levels
    :param median: median of the fragility function
    :param beta:   log std for the fragility function

    :return: the total risk and the disagg curve
    '''

    pdf_limitstate_im = stats.lognorm(beta, scale=median).pdf(imtl)
    disaggregation = pdf_limitstate_im * hcurve
    risk = np.trapz(disaggregation, x=imtl)

    return risk, disaggregation

