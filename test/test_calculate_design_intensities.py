import os
from unittest import TestCase
import pandas as pd
import numpy as np
import scipy
import risk_targeted_hazard as rth
from numpy.testing import assert_almost_equal


class TestCalculateDesignIntensities(TestCase):
    def generate_hcurves_test_data(self):
        # NOTE: The output of this function is cached in test/data/hcurves_test_data.npy
        # If you make changes you can delete the file to regenerate the test data
        hazard_id = "SLT_v8_gmm_v2_FINAL"

        res = next(
            rth.get_hazard_curves(
                ["-36.870~174.770"], [400], [hazard_id], ["PGA"], ["mean"]
            )
        )
        num_levels = len(res.values)

        site_codes = ["AKL", "WLG"]
        custom_location_dict = rth.LOCATIONS_BY_ID.copy()
        new_custom_location_dict = {}
        for site_code in site_codes:
            new_custom_location_dict[site_code] = custom_location_dict[site_code]
        custom_location_dict = new_custom_location_dict
        locations = [
            f"{loc['latitude']:0.3f}~{loc['longitude']:0.3f}"
            for loc in custom_location_dict.values()
        ]

        vs30_list = [250, 400]
        imts = ["SA(0.5)", "SA(1.5)"]
        aggs = ["mean", "0.1", "0.5", "0.9"]

        columns = ["lat", "lon", "vs30", "imt", "agg", "level", "hazard"]
        index = range(
            len(locations) * len(vs30_list) * len(imts) * len(aggs) * num_levels
        )
        pts_summary_data = pd.DataFrame(columns=columns, index=index)

        ind = 0
        for i, res in enumerate(
            rth.get_hazard_curves(locations, vs30_list, [hazard_id], imts, aggs)
        ):
            lat = f"{res.lat:0.3f}"
            lon = f"{res.lon:0.3f}"
            for value in res.values:
                pts_summary_data.loc[ind, "lat"] = lat
                pts_summary_data.loc[ind, "lon"] = lon
                pts_summary_data.loc[ind, "vs30"] = res.vs30
                pts_summary_data.loc[ind, "imt"] = res.imt
                pts_summary_data.loc[ind, "agg"] = res.agg
                pts_summary_data.loc[ind, "level"] = value.lvl
                pts_summary_data.loc[ind, "hazard"] = value.val
                ind += 1

        data_type = "points"
        summary_data = pts_summary_data
        vs30_list = list(np.unique(summary_data["vs30"].dropna()))

        site_list = sorted([loc["name"] for loc in custom_location_dict.values()])
        sites = pd.DataFrame(index=site_list, dtype="str")
        for loc in custom_location_dict.values():
            idx = (pts_summary_data["lat"].astype("float") == loc["latitude"]) & (
                pts_summary_data["lon"].astype("float") == loc["longitude"]
            )
            pts_summary_data.loc[idx, "name"] = loc["name"]
            sites.loc[loc["name"], ["lat", "lon"]] = [loc["latitude"], loc["longitude"]]

        imtl = list(np.unique(summary_data["level"].dropna()))
        imtls = {}
        for imt in imts:
            imtls[imt] = imtl

        data = {}
        data["metadata"] = {}
        data["metadata"]["quantiles"] = []
        data["metadata"]["acc_imtls"] = imtls

        data["metadata"]["vs30s"] = vs30_list

        idx_dict = {}

        idx_dict["vs30"] = {}
        for i_vs30, vs30 in enumerate(vs30_list):
            idx_dict["vs30"][vs30] = summary_data["vs30"] == vs30

        if data_type == "grid":
            idx_dict["lat"] = {}
            for i_lat, lat in enumerate(np.unique(sites["lat"])):
                idx_dict["lat"][lat] = summary_data["lat"] == f"{float(lat):.3f}"

            idx_dict["lon"] = {}
            for i_lon, lon in enumerate(np.unique(sites["lon"])):
                idx_dict["lon"][lon] = summary_data["lon"] == f"{float(lon):.3f}"
        else:
            idx_dict["site"] = {}
            for i_site, site in enumerate(sites.index):
                idx_dict["site"][site] = summary_data["name"] == site

        idx_dict["imt"] = {}
        for i_imt, imt in enumerate(imts):
            idx_dict["imt"][imt] = summary_data["imt"] == imt

        idx_dict["stat"] = {}
        for i_stat, stat in enumerate(aggs):
            idx_dict["stat"][stat] = summary_data["agg"] == stat

        hcurves = np.zeros(
            [len(vs30_list), len(locations), len(imts), len(imtl), len(aggs)]
        )

        for i_vs30, vs30 in enumerate(vs30_list):
            print(f"Vs30: {vs30}")
            vs30_idx = idx_dict["vs30"][vs30]

            if data_type == "grid":
                for i_latlon, latlon in enumerate(locations):
                    print(f"\tLat/Lon: {latlon}")
                    lat, lon = latlon.split("~")
                    latlon_idx = (idx_dict["lat"][lat]) & (idx_dict["lon"][lon])

                    for i_imt, imt in enumerate(imts):
                        print(
                            f"\t\tIMT: {imt}\t\t(Vs30: {vs30} at Lat/Lon: {latlon}  #{i_latlon+1} of {len(locations)})"
                        )
                        imt_idx = idx_dict["imt"][imt]

                        for i_stat, stat in enumerate(aggs):
                            print(f"\t\t\tStat: {stat}")
                            stat_idx = idx_dict["stat"][stat]

                            idx = vs30_idx & latlon_idx & imt_idx & stat_idx
                            hcurve = np.squeeze(summary_data[idx]["hazard"].to_numpy())

                            hcurves[i_vs30, i_latlon, i_imt, :, i_stat] = hcurve

            else:
                for i_site, site in enumerate(sites.index):
                    print(f"\tSite: {site}")
                    site_idx = idx_dict["site"][site]

                    for i_imt, imt in enumerate(imts):
                        print(
                            f"\t\tIMT: {imt}\t\t(Vs30: {vs30} at Site: {site}  #{i_site+1} of {len(sites)})"
                        )
                        imt_idx = idx_dict["imt"][imt]

                        for i_stat, stat in enumerate(aggs):
                            print(f"\t\t\tStat: {stat}")
                            stat_idx = idx_dict["stat"][stat]

                            idx = vs30_idx & site_idx & imt_idx & stat_idx
                            hcurve = np.squeeze(summary_data[idx]["hazard"].to_numpy())

                            hcurves[i_vs30, i_site, i_imt, :, i_stat] = hcurve

            data["hcurves"] = {}
            data["hcurves"]["hcurves_stats"] = hcurves

            return data

    def setUp(self):
        test_data_filename = os.path.dirname(__file__) + "/data/hcurves_test_data.npy"

        if os.path.isfile(test_data_filename):
            print(f"Loading existing hcurves test data from {test_data_filename}")
            self.data = np.load(test_data_filename, allow_pickle=True).item()
        else:
            print("Regenerating hcurves test data")
            self.data = self.generate_hcurves_test_data()
            np.save(test_data_filename, self.data)

    def test_calculate_hazard_design_intensities_acc_intensity_type(self):
        # Given
        intensity_type = "acc"
        hazard_rps = np.array([25, 50, 100, 250, 500, 1000, 2500])

        # When
        result = rth.calculate_hazard_design_intensities(
            self.data, hazard_rps, intensity_type
        )

        expected_result = np.array(
            [
                [
                    [
                        [
                            [
                                4.45348819e-02,
                                2.67933163e-02,
                                4.23534979e-02,
                                6.40153730e-02,
                            ],
                            [
                                8.01678833e-02,
                                5.09985646e-02,
                                7.62068442e-02,
                                1.10289964e-01,
                            ],
                            [
                                1.26821597e-01,
                                8.36212625e-02,
                                1.20779536e-01,
                                1.75050184e-01,
                            ],
                            [
                                2.19659679e-01,
                                1.40353501e-01,
                                2.07884954e-01,
                                2.99644808e-01,
                            ],
                            [
                                3.15087333e-01,
                                2.03049154e-01,
                                2.95305948e-01,
                                4.28368499e-01,
                            ],
                            [
                                4.37203557e-01,
                                2.77848794e-01,
                                4.06111505e-01,
                                5.95311982e-01,
                            ],
                            [
                                6.48877864e-01,
                                4.06165407e-01,
                                5.95303336e-01,
                                8.83035119e-01,
                            ],
                        ],
                        [
                            [
                                1.64268548e-02,
                                1.00821828e-02,
                                1.58200929e-02,
                                2.41335014e-02,
                            ],
                            [
                                3.33853512e-02,
                                2.18480373e-02,
                                3.20386179e-02,
                                4.72384867e-02,
                            ],
                            [
                                5.82956649e-02,
                                3.87051819e-02,
                                5.57847017e-02,
                                8.01217681e-02,
                            ],
                            [
                                1.03500371e-01,
                                6.88007200e-02,
                                9.87363694e-02,
                                1.37663651e-01,
                            ],
                            [
                                1.48123274e-01,
                                9.92634259e-02,
                                1.39032239e-01,
                                2.01836904e-01,
                            ],
                            [
                                2.10281924e-01,
                                1.34908775e-01,
                                1.95401667e-01,
                                2.80019057e-01,
                            ],
                            [
                                3.14993859e-01,
                                2.01822828e-01,
                                2.88727753e-01,
                                4.18139336e-01,
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                3.12253915e-01,
                                2.22447800e-01,
                                3.05887306e-01,
                                4.10304311e-01,
                            ],
                            [
                                5.06378474e-01,
                                3.53711814e-01,
                                4.93276590e-01,
                                6.72743184e-01,
                            ],
                            [
                                7.78499768e-01,
                                5.34680217e-01,
                                7.49966716e-01,
                                1.03348509e00,
                            ],
                            [
                                1.25713287e00,
                                8.50552487e-01,
                                1.19646783e00,
                                1.65262243e00,
                            ],
                            [
                                1.70456898e00,
                                1.14144067e00,
                                1.60839522e00,
                                2.21245720e00,
                            ],
                            [
                                2.22307538e00,
                                1.47703539e00,
                                2.07904767e00,
                                2.85540017e00,
                            ],
                            [
                                3.01546032e00,
                                1.98965060e00,
                                2.78995299e00,
                                3.83392522e00,
                            ],
                        ],
                        [
                            [
                                9.69158664e-02,
                                6.56859830e-02,
                                9.40806921e-02,
                                1.35021369e-01,
                            ],
                            [
                                1.77414635e-01,
                                1.15325519e-01,
                                1.69854495e-01,
                                2.56139994e-01,
                            ],
                            [
                                3.08137910e-01,
                                1.94626866e-01,
                                2.90284365e-01,
                                4.45186701e-01,
                            ],
                            [
                                5.64555726e-01,
                                3.48256232e-01,
                                5.18805146e-01,
                                7.98714237e-01,
                            ],
                            [
                                8.27047979e-01,
                                5.04101943e-01,
                                7.44757155e-01,
                                1.15079814e00,
                            ],
                            [
                                1.15051597e00,
                                6.88243491e-01,
                                1.01312159e00,
                                1.57894129e00,
                            ],
                            [
                                1.68130082e00,
                                9.75195406e-01,
                                1.42861846e00,
                                2.28401998e00,
                            ],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                        ],
                        [
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                        ],
                        [
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                            [
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                                1.00000000e-04,
                            ],
                        ],
                    ],
                ],
            ]
        )

        # Then
        assert_almost_equal(expected_result, result)

    def test_calculate_hazard_design_intensities_disp_intensity_type(self):
        # Given
        intensity_type = "disp"
        hazard_rps = np.array([25, 50, 100, 250, 500, 1000, 2500])
        self.data["metadata"]["disp_imtls"] = rth.convert_imtls_to_disp(
            self.data["metadata"]["acc_imtls"]
        )

        # TODO: investigate divide by zero warning
        # prepare_design_intensities.py:31: RuntimeWarning: divide by zero encountered in log
        # stats_im_hazard[i_vs30, i_site, i_imt, :, i_stat] = np.exp(np.interp(np.log(1 / hazard_rps), np.log(

        # When
        result = rth.calculate_hazard_design_intensities(
            self.data, hazard_rps, intensity_type
        )

        expected_result = np.array(
            [
                [
                    [
                        [
                            [
                                2.76567569e-03,
                                1.66390076e-03,
                                2.63020883e-03,
                                3.97544023e-03,
                            ],
                            [
                                4.97853271e-03,
                                3.16707903e-03,
                                4.73254688e-03,
                                6.84915420e-03,
                            ],
                            [
                                7.87579066e-03,
                                5.19299217e-03,
                                7.50057064e-03,
                                1.08708504e-02,
                            ],
                            [
                                1.36411596e-02,
                                8.71614002e-03,
                                1.29099335e-02,
                                1.86083431e-02,
                            ],
                            [
                                1.95673445e-02,
                                1.26096239e-02,
                                1.83388941e-02,
                                2.66022563e-02,
                            ],
                            [
                                2.71509253e-02,
                                1.72547815e-02,
                                2.52200673e-02,
                                3.69696698e-02,
                            ],
                            [
                                4.02961827e-02,
                                2.52234146e-02,
                                3.69691328e-02,
                                5.48376611e-02,
                            ],
                        ],
                        [
                            [
                                9.18116674e-03,
                                5.63505327e-03,
                                8.84204025e-03,
                                1.34885043e-02,
                            ],
                            [
                                1.86594744e-02,
                                1.22111309e-02,
                                1.79067690e-02,
                                2.64021585e-02,
                            ],
                            [
                                3.25821483e-02,
                                2.16327918e-02,
                                3.11787409e-02,
                                4.47810200e-02,
                            ],
                            [
                                5.78476023e-02,
                                3.84535501e-02,
                                5.51849446e-02,
                                7.69418707e-02,
                            ],
                            [
                                8.27878794e-02,
                                5.54795228e-02,
                                7.77067910e-02,
                                1.12809074e-01,
                            ],
                            [
                                1.17529097e-01,
                                7.54021373e-02,
                                1.09212342e-01,
                                1.56506019e-01,
                            ],
                            [
                                1.76053857e-01,
                                1.12801206e-01,
                                1.61373415e-01,
                                2.33703105e-01,
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                1.93913855e-02,
                                1.38143057e-02,
                                1.89960106e-02,
                                2.54804461e-02,
                            ],
                            [
                                3.14467801e-02,
                                2.19659764e-02,
                                3.06331356e-02,
                                4.17782509e-02,
                            ],
                            [
                                4.83458762e-02,
                                3.32043561e-02,
                                4.65739354e-02,
                                6.41808055e-02,
                            ],
                            [
                                7.80696316e-02,
                                5.28204463e-02,
                                7.43022514e-02,
                                1.02630062e-01,
                            ],
                            [
                                1.05856012e-01,
                                7.08849913e-02,
                                9.98834929e-02,
                                1.37396549e-01,
                            ],
                            [
                                1.38055952e-01,
                                9.17258722e-02,
                                1.29111639e-01,
                                1.77324167e-01,
                            ],
                            [
                                1.87264116e-01,
                                1.23559962e-01,
                                1.73259810e-01,
                                2.38091880e-01,
                            ],
                        ],
                        [
                            [
                                5.41674436e-02,
                                3.67126861e-02,
                                5.25828306e-02,
                                7.54650671e-02,
                            ],
                            [
                                9.91591740e-02,
                                6.44568198e-02,
                                9.49337205e-02,
                                1.43159724e-01,
                            ],
                            [
                                1.72221985e-01,
                                1.08779297e-01,
                                1.62243424e-01,
                                2.48820202e-01,
                            ],
                            [
                                3.15536987e-01,
                                1.94644597e-01,
                                2.89966437e-01,
                                4.46410995e-01,
                            ],
                            [
                                4.62247064e-01,
                                2.81748640e-01,
                                4.16253733e-01,
                                6.43194926e-01,
                            ],
                            [
                                6.43037218e-01,
                                3.84667566e-01,
                                5.66245844e-01,
                                8.82489279e-01,
                            ],
                            [
                                9.39699252e-01,
                                5.45048444e-01,
                                7.98472040e-01,
                                1.27656624e00,
                            ],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                        ],
                        [
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                            [
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                                6.21013366e-06,
                            ],
                        ],
                        [
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                            [
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                                5.58912029e-05,
                            ],
                        ],
                    ],
                ],
            ]
        )

        # Then
        assert_almost_equal(expected_result, result)

    def test_calculate_risk_design_intensities(self):
        # Given
        R_assumptions = {}

        baseline_rp = 500
        baseline_risk = 1.5e-5
        cmr = 4

        beta = 0.45
        design_point = scipy.stats.lognorm(beta, scale=cmr).cdf(1)
        p_fatality_given_collapse = 0.1

        R_assumptions["baseline_risk"] = baseline_risk
        R_assumptions["R_rps"] = [500, 1000, 2500]
        for R_rp in R_assumptions["R_rps"]:
            risk_factor = baseline_rp / R_rp
            fatality_risk_target = round(baseline_risk * risk_factor, 12)
            R_assumptions[f"APoE: 1/{R_rp}"] = {
                "risk_factor": risk_factor,
                "fatality_risk_target": fatality_risk_target,
            }

        risk_assumptions = {}
        for R_rp in R_assumptions["R_rps"]:
            key = f"APoE: 1/{R_rp}"
            risk_assumptions[key] = {
                "risk_factor": R_assumptions[f"APoE: 1/{R_rp}"]["risk_factor"],
                "fatality_risk_target": R_assumptions[f"APoE: 1/{R_rp}"]["fatality_risk_target"],
                "R_rp": R_rp,
                "ls_risk_target": round(
                    R_assumptions[f"APoE: 1/{R_rp}"]["fatality_risk_target"]
                    / p_fatality_given_collapse,
                    12,
                ),
                "cmr": cmr,
                "beta": beta,
                "design_point": design_point,
                "p_fatality_given_collapse": p_fatality_given_collapse,
            }

        self.data["risk_design"] = {}

        intensity_type = "acc"
        self.data["risk_design"][intensity_type] = {}
        self.data["risk_design"][intensity_type]["risk_assumptions"] = risk_assumptions

        # When
        result = rth.calculate_risk_design_intensities(self.data, risk_assumptions)

        expected_result = {
            "im_risk": np.array(
                [
                    [
                        [
                            [0.30241699, 0.39370117, 0.54467773],
                            [0.14775391, 0.19511719, 0.27575684],
                        ],
                        [
                            [1.36123047, 1.67695313, 2.15910645],
                            [0.76811523, 0.98371582, 1.32646484],
                        ],
                    ],
                    [
                        [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
                        [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
                    ],
                ]
            ),
            "lambda_risk": np.array(
                [
                    [
                        [
                            [0.00219056, 0.001281, 0.00062717],
                            [0.00276802, 0.00127321, 0.00061251],
                        ],
                        [
                            [0.00341577, 0.00211001, 0.00109212],
                            [0.00233085, 0.0014142, 0.00073667],
                        ],
                    ],
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                ]
            ),
            "fragility_risk": np.array(
                [
                    [
                        [
                            [1.20966797, 1.57480469, 2.17871094],
                            [0.59101563, 0.78046875, 1.10302734],
                        ],
                        [
                            [5.44492188, 6.7078125, 8.63642578],
                            [3.07246094, 3.93486328, 5.30585938],
                        ],
                    ],
                    [
                        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                    ],
                ]
            ),
            "disagg_risk": np.array(
                [
                    [
                        [
                            [
                                [
                                    2.08560387e-092,
                                    3.00438056e-079,
                                    4.03522199e-067,
                                    1.66986987e-060,
                                    5.05282849e-056,
                                    1.13991691e-052,
                                    6.17815016e-043,
                                    3.04958181e-034,
                                    1.19592295e-029,
                                    1.29328462e-026,
                                    2.15977473e-024,
                                    3.16115876e-018,
                                    3.42932952e-013,
                                    9.01295597e-011,
                                    2.73053020e-009,
                                    2.82327325e-008,
                                    7.11934730e-006,
                                    5.38433289e-005,
                                    1.31263195e-004,
                                    1.91572281e-004,
                                    2.12640637e-004,
                                    2.00990584e-004,
                                    1.71856025e-004,
                                    1.37685763e-004,
                                    1.05632568e-004,
                                    5.74906936e-005,
                                    2.96135352e-005,
                                    1.49171394e-005,
                                    7.47495676e-006,
                                    3.76096716e-006,
                                    1.91000690e-006,
                                    9.81895346e-007,
                                    5.11722078e-007,
                                    2.70512441e-007,
                                    1.45064390e-007,
                                    3.24898258e-008,
                                    7.91050131e-009,
                                    2.07636617e-009,
                                    5.82754047e-010,
                                    5.48368080e-011,
                                    6.29372599e-012,
                                    8.49418824e-013,
                                    1.31479146e-013,
                                    2.28645835e-014,
                                ],
                                [
                                    8.44027992e-098,
                                    2.99926462e-084,
                                    9.93715553e-072,
                                    6.97367085e-065,
                                    3.06947448e-060,
                                    9.26065870e-057,
                                    1.23811778e-046,
                                    1.50757387e-037,
                                    1.00259533e-032,
                                    1.57712909e-029,
                                    3.52225502e-027,
                                    1.27172575e-020,
                                    3.40322913e-015,
                                    1.51681768e-012,
                                    6.68442062e-011,
                                    9.24291402e-010,
                                    5.74951212e-007,
                                    7.37406360e-006,
                                    2.61498137e-005,
                                    5.10384542e-005,
                                    7.18383015e-005,
                                    8.30029092e-005,
                                    8.44549690e-005,
                                    7.88829220e-005,
                                    6.94220599e-005,
                                    4.79117633e-005,
                                    3.01677328e-005,
                                    1.80834299e-005,
                                    1.05642204e-005,
                                    6.09725016e-006,
                                    3.50581769e-006,
                                    2.01857375e-006,
                                    1.16760572e-006,
                                    6.79789535e-007,
                                    3.98822315e-007,
                                    1.09187728e-007,
                                    3.16353973e-008,
                                    9.68069952e-009,
                                    3.11669137e-009,
                                    3.71899518e-010,
                                    5.21757969e-011,
                                    8.37965200e-012,
                                    1.51214864e-012,
                                    3.01652279e-013,
                                ],
                                [
                                    1.21709867e-104,
                                    1.31379265e-090,
                                    1.32226113e-077,
                                    1.77740424e-070,
                                    1.24068713e-065,
                                    5.35284058e-062,
                                    2.17393969e-051,
                                    8.04094774e-042,
                                    1.02429236e-036,
                                    2.55528286e-033,
                                    8.16087488e-031,
                                    8.95060704e-024,
                                    7.27601054e-018,
                                    6.21162362e-015,
                                    4.34119209e-013,
                                    8.58417036e-012,
                                    1.62204729e-008,
                                    3.98482284e-007,
                                    2.24101220e-006,
                                    6.25485601e-006,
                                    1.17923754e-005,
                                    1.74442574e-005,
                                    2.19858840e-005,
                                    2.48026367e-005,
                                    2.58440399e-005,
                                    2.38907894e-005,
                                    1.92594973e-005,
                                    1.43002173e-005,
                                    1.00900988e-005,
                                    6.89509135e-006,
                                    4.61898674e-006,
                                    3.05756762e-006,
                                    2.01071444e-006,
                                    1.31831700e-006,
                                    8.63883305e-007,
                                    3.02805241e-007,
                                    1.08673188e-007,
                                    4.01653903e-008,
                                    1.53104196e-008,
                                    2.44705849e-009,
                                    4.39543391e-010,
                                    8.74416210e-011,
                                    1.90582421e-011,
                                    4.50135236e-012,
                                ],
                            ],
                            [
                                [
                                    1.62111092e-078,
                                    2.01132480e-066,
                                    2.32043729e-055,
                                    2.27066077e-049,
                                    2.45298897e-045,
                                    2.47361416e-042,
                                    1.04276020e-033,
                                    3.65958217e-026,
                                    2.98023491e-022,
                                    1.05608576e-019,
                                    7.44827504e-018,
                                    7.72169700e-013,
                                    6.12523670e-009,
                                    3.48117455e-007,
                                    3.53901538e-006,
                                    1.56230026e-005,
                                    2.73270698e-004,
                                    4.27712365e-004,
                                    3.40266102e-004,
                                    2.08768699e-004,
                                    1.14560931e-004,
                                    5.99257131e-005,
                                    3.08086913e-005,
                                    1.58135255e-005,
                                    8.17202344e-006,
                                    2.26152624e-006,
                                    6.62589182e-007,
                                    2.05895794e-007,
                                    6.76657847e-008,
                                    2.34130170e-008,
                                    8.49202702e-009,
                                    3.21619168e-009,
                                    1.26696126e-009,
                                    5.17397088e-010,
                                    2.18348895e-010,
                                    2.86608968e-011,
                                    4.40034963e-012,
                                    7.69773723e-013,
                                    1.50107771e-013,
                                    7.49189191e-015,
                                    5.07627940e-016,
                                    4.33895847e-017,
                                    4.47666356e-018,
                                    5.40407179e-019,
                                ],
                                [
                                    8.87430459e-084,
                                    2.85198571e-071,
                                    8.52272567e-060,
                                    1.45529621e-053,
                                    2.33371493e-049,
                                    3.19706206e-046,
                                    3.49097782e-037,
                                    3.17349402e-029,
                                    4.50969476e-025,
                                    2.37218690e-022,
                                    2.27285809e-020,
                                    6.10341511e-015,
                                    1.25408382e-010,
                                    1.24371238e-008,
                                    1.87685049e-007,
                                    1.12558586e-006,
                                    5.09977161e-005,
                                    1.39283585e-004,
                                    1.64482570e-004,
                                    1.37098746e-004,
                                    9.66336096e-005,
                                    6.24639361e-005,
                                    3.85760703e-005,
                                    2.32761335e-005,
                                    1.39008461e-005,
                                    4.94125009e-006,
                                    1.78897394e-006,
                                    6.67783595e-007,
                                    2.57985202e-007,
                                    1.03160289e-007,
                                    4.26483987e-008,
                                    1.82020651e-008,
                                    8.00338737e-009,
                                    3.61848845e-009,
                                    1.67879102e-009,
                                    2.72307676e-010,
                                    5.02210618e-011,
                                    1.03275948e-011,
                                    2.32738793e-012,
                                    1.49204073e-013,
                                    1.24927820e-014,
                                    1.28270780e-015,
                                    1.55572955e-016,
                                    2.17035237e-017,
                                ],
                                [
                                    1.48160310e-090,
                                    1.55588369e-077,
                                    1.51928947e-065,
                                    5.18585050e-059,
                                    1.35938371e-054,
                                    2.72640975e-051,
                                    9.72790644e-042,
                                    2.88962985e-033,
                                    8.20839512e-029,
                                    7.05806528e-026,
                                    9.90044902e-024,
                                    8.68735792e-018,
                                    5.83275331e-013,
                                    1.15630812e-010,
                                    2.85239033e-009,
                                    2.50440009e-008,
                                    3.70772952e-006,
                                    2.02424851e-005,
                                    3.90759404e-005,
                                    4.76835784e-005,
                                    4.58906629e-005,
                                    3.85999577e-005,
                                    2.99460658e-005,
                                    2.20959940e-005,
                                    1.57982445e-005,
                                    7.66770153e-006,
                                    3.61238251e-006,
                                    1.69390781e-006,
                                    8.00257465e-007,
                                    3.83100201e-007,
                                    1.86385148e-007,
                                    9.22956568e-008,
                                    4.65282315e-008,
                                    2.38753168e-008,
                                    1.24624389e-008,
                                    2.63043540e-009,
                                    6.09421759e-010,
                                    1.53254123e-010,
                                    4.13472517e-011,
                                    3.61925613e-012,
                                    3.94329398e-013,
                                    5.08618365e-014,
                                    7.54361152e-015,
                                    1.25991242e-015,
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    1.65964915e-124,
                                    4.11944971e-109,
                                    9.53348286e-095,
                                    8.02088710e-087,
                                    2.05702943e-081,
                                    2.43529366e-077,
                                    2.27957136e-065,
                                    1.97192518e-054,
                                    1.61867843e-048,
                                    1.53626815e-044,
                                    1.39676855e-041,
                                    4.17930506e-033,
                                    9.88311690e-026,
                                    6.14579252e-022,
                                    1.76437456e-019,
                                    1.04742332e-017,
                                    6.27143865e-013,
                                    1.22167265e-010,
                                    3.07075719e-009,
                                    2.78038311e-008,
                                    1.38262978e-007,
                                    4.66439433e-007,
                                    1.20322792e-006,
                                    2.55509240e-006,
                                    4.68792675e-006,
                                    1.15157449e-005,
                                    2.11365340e-005,
                                    3.18781433e-005,
                                    4.18579926e-005,
                                    4.96721207e-005,
                                    5.46351332e-005,
                                    5.66968698e-005,
                                    5.62267529e-005,
                                    5.37959150e-005,
                                    5.00146851e-005,
                                    3.80156237e-005,
                                    2.64343234e-005,
                                    1.73969210e-005,
                                    1.10620518e-005,
                                    4.23201933e-006,
                                    1.57235802e-006,
                                    5.82504180e-007,
                                    2.17696277e-007,
                                    8.27921450e-008,
                                ],
                                [
                                    1.97219312e-129,
                                    9.99665681e-114,
                                    4.72443191e-099,
                                    6.03541307e-091,
                                    2.08171339e-085,
                                    3.10138736e-081,
                                    5.92843815e-069,
                                    1.04727313e-057,
                                    1.30531912e-051,
                                    1.66616874e-047,
                                    1.90634125e-044,
                                    1.16483041e-035,
                                    5.62516135e-028,
                                    5.31135755e-024,
                                    2.05075730e-021,
                                    1.53203950e-019,
                                    1.87325684e-014,
                                    5.54078876e-012,
                                    1.87308744e-010,
                                    2.13423119e-009,
                                    1.28057516e-008,
                                    5.06354190e-008,
                                    1.49879731e-007,
                                    3.59329084e-007,
                                    7.34852311e-007,
                                    2.17807900e-006,
                                    4.68570936e-006,
                                    8.10905875e-006,
                                    1.20211657e-005,
                                    1.59006316e-005,
                                    1.92935038e-005,
                                    2.18989405e-005,
                                    2.35838378e-005,
                                    2.43541682e-005,
                                    2.43100473e-005,
                                    2.16576019e-005,
                                    1.72803406e-005,
                                    1.28394745e-005,
                                    9.10005532e-006,
                                    4.20066776e-006,
                                    1.82928812e-006,
                                    7.77616689e-007,
                                    3.28101553e-007,
                                    1.39084875e-007,
                                ],
                                [
                                    1.59583797e-135,
                                    1.92120412e-119,
                                    2.15649122e-104,
                                    4.56944490e-096,
                                    2.25682773e-090,
                                    4.44196602e-086,
                                    2.01668880e-073,
                                    8.46131621e-062,
                                    1.74925454e-055,
                                    3.19724797e-051,
                                    4.83281009e-048,
                                    7.01360262e-039,
                                    8.04438439e-031,
                                    1.25985834e-026,
                                    6.96548673e-024,
                                    6.87462338e-022,
                                    1.99643718e-016,
                                    9.79463510e-014,
                                    4.74128158e-012,
                                    7.13708556e-011,
                                    5.37651814e-010,
                                    2.57690670e-009,
                                    9.01072994e-009,
                                    2.50234253e-008,
                                    5.83658585e-008,
                                    2.17194610e-007,
                                    5.66368391e-007,
                                    1.15788889e-006,
                                    1.98829281e-006,
                                    2.99952553e-006,
                                    4.09927006e-006,
                                    5.18654190e-006,
                                    6.17236608e-006,
                                    6.99159614e-006,
                                    7.60645564e-006,
                                    8.21400844e-006,
                                    7.74229756e-006,
                                    6.66349576e-006,
                                    5.38645874e-006,
                                    3.12172211e-006,
                                    1.64780670e-006,
                                    8.27489527e-007,
                                    4.04429230e-007,
                                    1.95532334e-007,
                                ],
                            ],
                            [
                                [
                                    1.78407664e-111,
                                    6.24570553e-097,
                                    2.03636744e-083,
                                    5.42986000e-076,
                                    6.14229894e-071,
                                    3.84169168e-067,
                                    4.79948965e-056,
                                    5.11806907e-046,
                                    1.17549442e-040,
                                    4.41794884e-037,
                                    1.93573841e-034,
                                    5.77701614e-027,
                                    1.39822900e-020,
                                    2.42810313e-017,
                                    2.92239536e-015,
                                    9.02132392e-014,
                                    7.66801148e-010,
                                    4.82111700e-008,
                                    5.34769192e-007,
                                    2.53380789e-006,
                                    7.35993136e-006,
                                    1.56778900e-005,
                                    2.70743561e-005,
                                    4.02866438e-005,
                                    5.37289804e-005,
                                    7.59555271e-005,
                                    8.74502228e-005,
                                    8.81975827e-005,
                                    8.13522748e-005,
                                    7.05158756e-005,
                                    5.84754956e-005,
                                    4.69614696e-005,
                                    3.68413631e-005,
                                    2.84085408e-005,
                                    2.16302166e-005,
                                    1.05670606e-005,
                                    5.03418485e-006,
                                    2.38203354e-006,
                                    1.13087715e-006,
                                    2.62807316e-007,
                                    6.44286465e-008,
                                    1.67237318e-008,
                                    4.59441651e-009,
                                    1.32736236e-009,
                                ],
                                [
                                    5.05069973e-117,
                                    4.12375076e-102,
                                    3.13573902e-088,
                                    1.37216309e-080,
                                    2.20590893e-075,
                                    1.81207243e-071,
                                    5.27984249e-060,
                                    1.31312150e-049,
                                    4.94940139e-044,
                                    2.64357913e-040,
                                    1.52129948e-037,
                                    1.05887451e-029,
                                    5.97712141e-023,
                                    1.70339055e-019,
                                    2.91357100e-017,
                                    1.18128130e-015,
                                    2.34173997e-011,
                                    2.41622247e-009,
                                    3.80886129e-008,
                                    2.37027577e-007,
                                    8.60272171e-007,
                                    2.21227760e-006,
                                    4.49738198e-006,
                                    7.72781171e-006,
                                    1.17221333e-005,
                                    2.07059115e-005,
                                    2.87796687e-005,
                                    3.41689146e-005,
                                    3.63946885e-005,
                                    3.58804450e-005,
                                    3.34283464e-005,
                                    2.98572300e-005,
                                    2.58293164e-005,
                                    2.18045264e-005,
                                    1.80619701e-005,
                                    1.06524216e-005,
                                    5.97410620e-006,
                                    3.26426536e-006,
                                    1.76260756e-006,
                                    5.11816831e-007,
                                    1.51476764e-007,
                                    4.62860197e-008,
                                    1.46838795e-008,
                                    4.82505821e-009,
                                ],
                                [
                                    6.67381335e-124,
                                    1.51601326e-108,
                                    3.20729262e-094,
                                    2.55361118e-086,
                                    6.27732098e-081,
                                    7.16841797e-077,
                                    5.81107952e-065,
                                    4.02095162e-054,
                                    2.75757274e-048,
                                    2.25218641e-044,
                                    1.80172185e-041,
                                    3.48903724e-033,
                                    5.47950432e-026,
                                    2.84127844e-022,
                                    7.43126734e-020,
                                    4.18843345e-018,
                                    2.31006948e-013,
                                    4.33684257e-011,
                                    1.04536899e-009,
                                    9.04345422e-009,
                                    4.29596463e-008,
                                    1.38705278e-007,
                                    3.43417417e-007,
                                    7.02152941e-007,
                                    1.24431462e-006,
                                    2.87678404e-006,
                                    5.02027189e-006,
                                    7.25908378e-006,
                                    9.20028494e-006,
                                    1.05966711e-005,
                                    1.13640060e-005,
                                    1.15411877e-005,
                                    1.12364882e-005,
                                    1.05822055e-005,
                                    9.70570573e-006,
                                    7.18685512e-006,
                                    4.90876971e-006,
                                    3.19152184e-006,
                                    2.01333645e-006,
                                    7.65181827e-007,
                                    2.84331283e-007,
                                    1.05812662e-007,
                                    3.99430532e-008,
                                    1.53338556e-008,
                                ],
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                            ],
                            [
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                            ],
                            [
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ],
                            ],
                        ],
                    ],
                ]
            ),
        }

        # Then
        with self.subTest("has expected risk types"):
            self.assertCountEqual(expected_result.keys(), result.keys())

        with self.subTest("has expected im_risk"):
            assert_almost_equal(expected_result["im_risk"], result["im_risk"])

        with self.subTest("has expected lambda_risk"):
            assert_almost_equal(expected_result["lambda_risk"], result["lambda_risk"])

        with self.subTest("has expected fragility_risk"):
            assert_almost_equal(
                expected_result["fragility_risk"], result["fragility_risk"]
            )

        with self.subTest("has expected disagg_risk"):
            assert_almost_equal(expected_result["disagg_risk"], result["disagg_risk"])

    def test_imtl_lognorm_pdf(self):
        # Given
        beta = 0.45
        median = 0.5
        imtl = np.array([
            0.0001, 0.0002, 0.0004,0.0006,
            0.0008, 0.001, 0.002, 0.004,
            0.006, 0.008, 0.01, 0.02,
            0.04, 0.06, 0.08, 0.1,
            0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9,
            1, 1.2, 1.4, 1.6,
            1.8, 2, 2.2, 2.4,
            2.6, 2.8, 3, 3.5,
            4, 4.5, 5, 6,
            7, 8, 9, 10
        ])

        # When
        result = rth.imtl_lognorm_pdf(beta, median, imtl)

        # Then
        expected_result = np.array([
            1.43867893e-74, 1.00719728e-62, 6.57438103e-52, 4.63877416e-46,
            4.00115028e-42, 3.41021596e-39, 9.01536893e-31, 2.22215726e-23,
            1.55964702e-19, 5.10688107e-17, 3.44198758e-15, 3.43606904e-10,
            3.19818902e-06, 2.23284629e-04, 2.77546683e-03, 1.47926552e-02,
            5.57634552e-01, 1.55152561e+00, 1.95994048e+00, 1.77307680e+00,
            1.36113359e+00, 9.57628309e-01, 6.42282007e-01, 4.19731583e-01,
            2.70702574e-01, 1.11335129e-01, 4.62139398e-02, 1.96244862e-02,
            8.56942554e-03, 3.85342479e-03, 1.78378703e-03, 8.49089265e-04,
            4.14997015e-04, 2.07940305e-04, 1.06649625e-04, 2.20380458e-05,
            5.11435718e-06, 1.31064418e-06, 3.65874596e-07, 3.53306878e-08,
            4.30735229e-09, 6.32885125e-10, 1.08374264e-10, 2.10934712e-11
        ])
        assert_almost_equal(expected_result, result)

    def test_imtl_lognorm_pdf_divide_by_zero_handling(self):
        # Given
        beta = 0.45
        median = 0.5
        imtl = np.array([0.0, 1.0])

        # When
        result = rth.imtl_lognorm_pdf(beta, median, imtl)

        # Then
        expected_result = np.array([0., 0.27070257])
        assert_almost_equal(expected_result, result)
