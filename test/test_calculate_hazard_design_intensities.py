import json
import os
from unittest import TestCase
import pandas as pd
import numpy as np
import risk_targeted_hazard as rth
from numpy.testing import assert_almost_equal


class TestCalculateHazardDesignIntensities(TestCase):
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
        #prepare_design_intensities.py:31: RuntimeWarning: divide by zero encountered in log
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
