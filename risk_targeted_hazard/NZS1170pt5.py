from .base import *

# Near fault factor interpolation
near_fault_T = np.array([1.5,2,3,4,5])
N_max_T = np.array([1,1.12,1.36,1.60,1.72])


# R factors for return period adjustments
NZS1170_R = np.array([[20,  25,   50,   100, 250,  500, 1000, 2000, 2500],
                      [0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.3,  1.7,  1.8]])
NZS1170_R_dict = {}
for rp,R in zip(NZS1170_R[0,:],NZS1170_R[1,:]):
    NZS1170_R_dict[rp] = R


# design spectral shape
NZS1170_T = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
Ch = pd.DataFrame(index=NZS1170_T)
Ch['A'] = [1.00, 2.35, 2.35, 2.35, 1.89, 1.60, 1.40, 1.24, 1.12, 1.03, 0.95, 0.70, 0.53, 0.42, 0.35, 0.26, 0.20, 0.16]
Ch['B'] = Ch['A'].values
Ch['C'] = [1.33, 2.93, 2.93, 2.93, 2.36, 2.00, 1.74, 1.55, 1.41, 1.29, 1.19, 0.88, 0.66, 0.53, 0.44, 0.32, 0.25, 0.20]
Ch['D'] = [1.12, 3.00, 3.00, 3.00, 3.00, 3.00, 2.84, 2.53, 2.29, 2.09, 1.93, 1.43, 1.07, 0.86, 0.71, 0.52, 0.40, 0.32]
Ch['E'] = [3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 2.21, 1.66, 1.33, 1.11, 0.81, 0.62, 0.49]

# Z values from NZS1170.5
site_Z_values= {'Te Anau': 0.36,
 'Invercargill': 0.17,
 'Queenstown': 0.32,
 'Haast': np.nan,
 'Mount Cook': 0.38,
 'Franz Josef': 0.44,
 'Dunedin': 0.13,
 'Greymouth': 0.37,
 'Timaru': 0.15,
 'Otira': 0.6,
 'Westport': 0.3,
 'Christchurch': 0.3,
 'Hammer Springs': 0.55,
 'Nelson': 0.27,
 'Kaikoura': 0.42,
 'Blenheim': 0.33,
 'Kerikeri': 0.1,
 'New Plymouth': 0.18,
 'Hawera': 0.18,
 'Auckland': 0.13,
 'Wellington': 0.4,
 'Whanganui': np.nan,
 'Levin': 0.4,
 'Hamilton': 0.16,
 'Thames': 0.16,
 'Palmerston North': 0.38,
 'Masterton': 0.42,
 'Tokoroa': 0.21,
 'Turangi': 0.27,
 'Taupo': 0.28,
 'Tauranga': 0.2,
 'Napier': 0.38,
 'Whakatane': 0.3,
 'Gisborne': 0.36}