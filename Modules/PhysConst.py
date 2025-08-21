#########################################################################
## Dictionaries for Physical constants & other information
#########################################################################
##   List of the Standard Atomic Weight from 
##   https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
##   Access: 10.11.2022
##   
##   if a range [a,b] of Standard Atomic Weights is given we take the mean b+a/2 as the Standard Atomic Weight
##   if an uncertainty e.g. 28.976494700(22)is given in the Standard Atomic Weights, we take the mean.
StandardAtomicWeights={'Ac': 227.0,
 'Ag': 107.8682,
 'Al': 26.9815385,
 'Ar': 39.948,
 'As': 74.921595,
 'At': 210.0,
 'Au': 196.966569,
 'B': 10.8135,
 'Ba': 137.327,
 'Be': 9.0121831,
 'Bi': 208.9804,
 'Br': 79.904,
 'C': 12.0106,
 'Ca': 40.078,
 'Cd': 112.414,
 'Ce': 140.116,
 'Cl': 35.451499999999996,
 'Co': 58.933194,
 'Cr': 51.9961,
 'Cs': 132.90545196,
 'Cu': 63.546,
 'Dy': 162.5,
 'Er': 167.259,
 'Eu': 151.964,
 'F': 18.998403163,
 'Fe': 55.845,
 'Fr': 223.0,
 'Ga': 69.723,
 'Gd': 157.25,
 'Ge': 72.63,
 'H': 1.007975,
 'He': 4.002602,
 'Hf': 178.49,
 'Hg': 200.592,
 'Ho': 164.93033,
 'I': 126.90447,
 'In': 114.818,
 'Ir': 192.217,
 'K': 39.0983,
 'Kr': 83.798,
 'La': 138.90547,
 'Li': 6.967499999999999,
 'Lu': 174.9668,
 'Mg': 24.3055,
 'Mn': 54.938044,
 'Mo': 95.95,
 'N': 14.006855,
 'Na': 22.98976928,
 'Nb': 92.90637,
 'Nd': 144.242,
 'Ne': 20.1797,
 'Ni': 58.6934,
 'Np': 237.0,
 'O': 15.9994,
 'Os': 190.23,
 'P': 30.973761998,
 'Pa': 231.03588,
 'Pb': 207.2,
 'Pd': 106.42,
 'Pm': 145.0,
 'Po': 209.0,
 'Pr': 140.90766,
 'Pt': 195.084,
 'Pu': 244.0,
 'Ra': 226.0,
 'Rb': 85.4678,
 'Re': 186.207,
 'Rh': 102.9055,
 'Rn': 222.0,
 'Ru': 101.07,
 'S': 32.067499999999995,
 'Sb': 121.76,
 'Sc': 44.955908,
 'Se': 78.971,
 'Si': 28.085,
 'Sm': 150.36,
 'Sn': 118.71,
 'Sr': 87.62,
 'Ta': 180.94788,
 'Tb': 158.92535,
 'Tc': 98.0,
 'Te': 127.6,
 'Th': 232.0377,
 'Ti': 47.867,
 'Tl': 204.3835,
 'Tm': 168.93422,
 'U': 238.02891,
 'V': 50.9415,
 'W': 183.84,
 'Xe': 131.293,
 'Y': 88.90584,
 'Yb': 173.054,
 'Zn': 65.38,
 'Zr': 91.224}


# --- Physical Constants ---
##   List of Physical constants in SI-units
##   https://physics.nist.gov/cuu/Constants
##   Access: 10.11.2022
##   
##   We take the numerical value for information on the Standard uncertainty and Relative Standard uncertainty 
##   consult https://physics.nist.gov/cuu/Constants
PhysicalConstants = {
    'm_u': 1.66053906660e-27,
    'm_e': 9.109383701510e-31,
    'm_p': 1.67262192369e-27,
    'N_A': 6.02214076e23,
    'k_B': 1.380649e-23,
    'h': 6.62607015e-34,
    'hbar': 1.054571817e-34,
    'e': 1.602176634e-19,
    'epsilon_0': 8.854878128e-12,
    'mu_0': 1.25663706212e-6,
}

# --- Conversion Factors ---
ConversionFactors = {
    'u->a.u.': 1.82288848426455e3,
    'a.u.->u': 1 / 1.82288848426455e3,
    'A->a.u.': 1.88972613288564,
    'a.u.->A': 1 / 1.88972613288564,
    'a.u.->eV': 2.72113838565563e1,
    'eV->a.u.': 1 / 2.72113838565563e1,
    'a.u.->1/cm': 2.19474631370540e5,
    '1/cm->a.u.': 1 / 2.19474631370540e5,
    '1/cm->J': 1.98644585714893e-23,
    'J->1/cm': 1 / 1.98644585714893e-23,
    'a.u.->N': 8.2387234983e-8,
    'N->a.u.': 1 / 8.2387234983e-8,
    'E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)': 1.7028697996e6,
}

# --- Atom Symbol to Number Mapping ---
AtomSymbolToAtomNumber = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Ni": 27, "Co": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "Ce": 58,
}
#########################################################################
## END Dictionaries for Physical constants & other Data
#########################################################################
