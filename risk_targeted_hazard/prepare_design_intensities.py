from .base import *
from numba import njit, typed, prange


@njit(parallel=True)
def _hazard_design_intensities_interpolate(
    hcurves_stats,
    stats_im_hazard,
    hazard_rps_reciprocal_log,
    imtls_imt_flip_logs,
    n_vs30,
    n_sites,
    n_imts,
    n_stats,
) -> None:
    for i_site in prange(n_sites):
        for i_vs30 in range(n_vs30):
            for i_imt in range(n_imts):
                # loop over the median and any quantiles
                for i_stat in range(n_stats):
                    # the interpolation is done as a linear interpolation in logspace
                    # all inputs are converted to the natural log (which is log in numpy) and the output is converted back via the exponent
                    stats_im_hazard[i_vs30, i_site, i_imt, :, i_stat] = np.exp(np.interp(hazard_rps_reciprocal_log, np.log(np.flip(hcurves_stats[i_vs30, i_site, i_imt, :, i_stat])), imtls_imt_flip_logs[i_imt]))

def calculate_hazard_design_intensities(data,hazard_rps,intensity_type='acc'):
    '''
    calculate design intensities based on an annual probability of exceedance (APoE)

    :param data: dictionary containing hazard curves and metadata for vs30, sites, intensity measures
    :param hazard_rps: np array containing the desired return periods (1 / APoE)

    :return: np arrays for all intensities from the hazard curve realizations and stats (mean and quantiles)
    '''

    vs30s = data['metadata']['vs30s']
    imtls = data['metadata'][f'{intensity_type}_imtls']
    hcurves_stats = np.array(data['hcurves']['hcurves_stats'])

    [n_vs30, n_sites, n_imts, n_imtls, n_stats] = hcurves_stats.shape

    n_rps = len(hazard_rps)

    stats_im_hazard = np.zeros([n_vs30, n_sites, n_imts, n_rps, n_stats])

    hazard_rps_reciprocal_log = np.log(1 / hazard_rps)

    imtls_imt_flip_logs = typed.List()
    for imt in imtls:
        imtls_imt_flip_logs.append(np.log(np.flip(imtls[imt])))

    _hazard_design_intensities_interpolate(
        hcurves_stats,
        stats_im_hazard,
        hazard_rps_reciprocal_log,
        imtls_imt_flip_logs,
        n_vs30,
        n_sites,
        n_imts,
        n_stats
    )

    return stats_im_hazard


def calculate_risk_design_intensities(data,risk_assumptions):
    '''
    calculate design intensities based on a risk target and fragility assumptions

    :param data: dictionary containing hazard curves and metadata for vs30, sites, intensity measures
    :param risk_target_assumptions: dictionary with keys for combinations of assumptions

    :return: np arrays for all intensities from the mean hazard curve
    '''

    intensity_type = 'acc'
    vs30s = data['metadata']['vs30s']
    imtls = data['metadata'][f'{intensity_type}_imtls']
    n_imtls = len(imtls[list(imtls.keys())[0]])
    hcurves_stats = np.array(data['hcurves']['hcurves_stats'])

    [n_vs30, n_sites, n_imts, n_imtls, n_stats] = hcurves_stats.shape

    n_risk_assumptions = len(risk_assumptions.keys())

    im_risk = np.zeros([n_vs30, n_sites, n_imts, n_risk_assumptions])
    lambda_risk = np.zeros_like(im_risk)
    fragility_risk = np.zeros_like(im_risk)
    disagg_risk = np.zeros([n_vs30, n_sites, n_imts, n_risk_assumptions, n_imtls])

    # select the statistic for the mean
    i_stat = 0

    for i_vs30, vs30 in enumerate(vs30s):
        print(f'Processing Vs30: {vs30}.')
        for i_imt, imt in enumerate(imtls.keys()):
            print(f'\tProcessing {imt}.')

            if imt != 'PGA':
                for i_site in range(n_sites):
                    # loop over the risk target assumption dictionaries
                    for i_rt, rt in enumerate(risk_assumptions.keys()):
                        collapse_risk_target = risk_assumptions[rt]['collapse_risk_target']
                        cmr = risk_assumptions[rt]['cmr']
                        beta = risk_assumptions[rt]['beta']
                        design_point = risk_assumptions[rt]['design_point']

                        # find the optimized fragility, defined by either the design point or the median
                        [im_r, median] = find_uniform_risk_intensity(hcurves_stats[i_vs30, i_site, i_imt, :, i_stat],
                                                                     imtls[imt], beta, collapse_risk_target,
                                                                     design_point)
                        # store the design intensity, defined by the design point of the optimized fragility
                        im_risk[i_vs30, i_site, i_imt, i_rt] = im_r
                        # store the probability of exceedance associated with the design intensity
                        lambda_risk[i_vs30, i_site, i_imt, i_rt] = np.interp(im_r, imtls[imt],
                                                                             hcurves_stats[i_vs30, i_site, i_imt, :,
                                                                             i_stat])
                        # store the median of the optimized fragility
                        fragility_risk[i_vs30, i_site, i_imt, i_rt] = median

                        # recalculate the risk value and the retrieve the risk integrand curve (i.e. the disaggregation)
                        risk, disagg = risk_convolution(hcurves_stats[i_vs30, i_site, i_imt, :, i_stat], imtls[imt],
                                                        median, beta)
                        # store the disaggregation
                        disagg_risk[i_vs30, i_site, i_imt, i_rt, :] = disagg

            else:
                # if the imt is PGA, then the period is 0 and the idea of a collapse fragility is meaningless
                # instead, the design intensity is based on the relevant probability of exceedance
                for i_site in range(n_sites):
                    # loop over the risk target assumption dictionaries
                    for i_rt, rt in enumerate(risk_assumptions.keys()):
                        hazard_rp = risk_assumptions[rt]['R_rp']
                        im_risk[i_vs30, i_site, i_imt, i_rt] = np.exp(np.interp(np.log(1 / hazard_rp), np.log(
                            np.flip(hcurves_stats[i_vs30, i_site, i_imt, :, i_stat])), np.log(np.flip(imtls[imt]))))
                        lambda_risk[i_vs30, i_site, i_imt, i_rt] = 1 / hazard_rp
                        fragility_risk[i_vs30, i_site, i_imt, i_rt] = np.nan
                        disagg_risk[i_vs30, i_site, i_imt, i_rt, :] = np.nan
        print()

    # store results as a dictionary
    im_risk = {'im_risk': im_risk, 'lambda_risk': lambda_risk, 'fragility_risk': fragility_risk,
               'disagg_risk': disagg_risk}
    return im_risk



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
    # the derivative of the fragility function, characterized as the pdf instead of the cdf
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

