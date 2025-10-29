# Import all Packages used by this module
import numpy as np
from .PhysConst import ConversionFactors
import os
from ctypes import c_char_p, cdll, POINTER, c_double, c_int, Structure
from copy import deepcopy

from ._extension import *

# get the environmental variable
pathtocp2k = os.environ["cp2kpath"]
# Get the directory of the current file (atomic_basis.py)

# TODO: Remove - no longer needed with Pybind11
_module_dir = os.path.dirname(__file__)
# Build the path to the .so relative to this module
# Normalize the path (removes .. etc.)
cpp_lib_path = os.path.join(_module_dir, "..", "CPP_Extension", "bin", "AtomicBasis.so")
pathtocpp_lib = os.path.abspath(cpp_lib_path)


# pathtocpp_lib="../CPP_Extension/bin/AtomicBasis.so"
# Python Implementation of the Overlap for Normalization of the Basis
def gamma(alpha, n):
    ## computes the analytical value of the integral int_{-\inf}^{inf}x^ne^{-alphax^2}
    ## (see Manuscript gamma function)
    ## input:       alpha       gaussian exponent                   (float)
    ##              n           power of x in the integral          (int)
    ## output:      value       the actual value of the integral    (float)
    def doublefactorial(n):
        if n == -1:
            return 1
        elif n == 0:
            return 1
        else:
            return n * doublefactorial(n - 2)

    value = 0.0
    if n % 2 == 0:
        value = (doublefactorial(n - 1) * np.sqrt(np.pi)) / (
            2 ** (0.5 * n) * alpha ** (0.5 * n + 0.5)
        )
    return value


# -------------------------------------------------------------------------
def Kcomponent(Y1k, Y2k, ik, jk, alpha):
    ## computes the analytical value of the integral int_{-\inf}^{inf}dx (x-Y1k)^ik (x-Y2k)^jk e^{-alpha x^2}
    ## for one component k
    ## (see Manuscript K-Function function)
    ## input:       alpha        gaussian exponent                  (float)
    ##              Y1k          displacement Y1k                   (float)
    ##              Y2k          displacement Y2k                   (float)
    ##              ik           power of x-Y1k in the integral     (int)
    ##              jk           power of x-Y2k in the integral     (int)
    ## output:      sum          the  value of the integral         (float)
    def binom(n, k):
        return np.math.factorial(n) / np.math.factorial(k) / np.math.factorial(n - k)

    sum = 0.0
    if Y1k == 0.0 or Y2k == 0.0:
        sum = gamma(alpha, ik + jk)
    else:
        for o in range(ik + 1):
            for p in range(jk + 1):
                if ik == o and jk == p:
                    sum += gamma(alpha, o + p)
                else:
                    sum += (
                        gamma(alpha, o + p)
                        * binom(ik, o)
                        * binom(jk, p)
                        * (-Y1k) ** (ik - o)
                        * (-Y2k) ** (jk - p)
                    )
    return sum


# -------------------------------------------------------------------------
def KFunction(Y1, Y2, iis, jjs, alpha):
    # the full K function iis=(ix,iy,iz) jjs=(jx,jy,jz)
    ## Computes the analytical value of full K function:
    ##  K=\prod_{k=1}^3int_{-\inf}^{inf}dx (x-Y1k)^ik (x-Y2k)^jk e^{-alpha x^2}
    ## (see Manuscript K-Function function)
    ## input:       alpha           gaussian exponent                  (float)
    ##              Y1              displacement Y1                    (np.array)
    ##              Y2k             displacement Y2                    (np.array)
    ##              iis=(ix,iy,iz)  monomial decomposition             (np.array)
    ##              jjs=(jx,jy,jz)  monomial decomposition             (np.array)
    ## output:      output             the  value of the integral      (float)
    return np.prod(
        [Kcomponent(Y1[it], Y2[it], iis[it], jjs[it], alpha) for it in range(len(Y1))]
    )


# -------------------------------------------------------------------------
def JInt(X, lm1, lm2, A1, A2):
    # computes the J integral using the monomial decomposition of the
    # solid harmonics.
    # Input: X (numpy.array) of the difference vector R1-R2
    # A1: positive numerical
    # A2: positive numerical

    ###############################################################################################################################
    ###############################################################################################################################
    # Define the cs hash map for the coefficients of the solid harmonics to homigenious monomials
    # returns the representation of a given solid harmonics (l,m) in terms of homogenious monomials
    # input:
    # format:
    # cs
    # is=cs[it][0] is the representation of the monomial in terms of
    # x^is[0]y^is[1]z^is[2]
    # and cs[it][1] the corresponding prefactor (consistent with CP2K convention)
    cs = {}

    cs["s"] = [[[0, 0, 0], 0.5 / np.sqrt(np.pi)]]

    cs["py"] = [[[0, 1, 0], np.sqrt(3.0 / (4.0 * np.pi))]]
    cs["pz"] = [[[0, 0, 1], np.sqrt(3.0 / (4.0 * np.pi))]]
    cs["px"] = [[[1, 0, 0], np.sqrt(3.0 / (4.0 * np.pi))]]

    cs["d-2"] = [[[1, 1, 0], 0.5 * np.sqrt(15.0 / np.pi)]]
    cs["d-1"] = [[[0, 1, 1], 0.5 * np.sqrt(15.0 / np.pi)]]
    cs["d0"] = [
        [[2, 0, 0], -0.25 * np.sqrt(5.0 / np.pi)],
        [[0, 2, 0], -0.25 * np.sqrt(5.0 / np.pi)],
        [[0, 0, 2], 0.5 * np.sqrt(5.0 / np.pi)],
    ]
    cs["d+1"] = [[[1, 0, 1], 0.5 * np.sqrt(15.0 / np.pi)]]
    cs["d+2"] = [
        [[2, 0, 0], 0.25 * np.sqrt(15.0 / np.pi)],
        [[0, 2, 0], -0.25 * np.sqrt(15.0 / np.pi)],
    ]

    cs["f-3"] = [
        [[2, 1, 0], 0.75 * np.sqrt(35.0 / 2.0 / np.pi)],
        [[0, 3, 0], -0.25 * np.sqrt(35.0 / 2.0 / np.pi)],
    ]
    cs["f-2"] = [[[1, 1, 1], 0.5 * np.sqrt(105.0 / np.pi)]]
    cs["f-1"] = [
        [[0, 1, 2], np.sqrt(21.0 / 2.0 / np.pi)],
        [[2, 1, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
        [[0, 3, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
    ]
    cs["f0"] = [
        [[0, 0, 3], 0.5 * np.sqrt(7.0 / np.pi)],
        [[2, 0, 1], -0.75 * np.sqrt(7 / np.pi)],
        [[0, 2, 1], -0.75 * np.sqrt(7 / np.pi)],
    ]
    cs["f+1"] = [
        [[1, 0, 2], np.sqrt(21.0 / 2.0 / np.pi)],
        [[1, 2, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
        [[3, 0, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
    ]
    cs["f+2"] = [
        [[2, 0, 1], 0.25 * np.sqrt(105.0 / np.pi)],
        [[0, 2, 1], -0.25 * np.sqrt(105.0 / np.pi)],
    ]
    cs["f+3"] = [
        [[3, 0, 0], 0.25 * np.sqrt(35.0 / 2.0 / np.pi)],
        [[1, 2, 0], -0.75 * np.sqrt(35.0 / 2.0 / np.pi)],
    ]

    cs["g-4"] = [
        [[3, 1, 0], 0.75 * np.sqrt(35.0 / np.pi)],
        [[1, 3, 0], -0.75 * np.sqrt(35.0 / np.pi)],
    ]
    cs["g-3"] = [
        [[2, 1, 1], 9.0 * np.sqrt(35.0 / (2 * np.pi)) / 4.0],
        [[0, 3, 1], -0.75 * np.sqrt(35.0 / (2.0 * np.pi))],
    ]
    cs["g-2"] = [
        [[1, 1, 2], 18.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
        [[3, 1, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
        [[1, 3, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
    ]
    cs["g-1"] = [
        [[0, 1, 3], 3.0 * np.sqrt(5.0 / (2 * np.pi))],
        [[2, 1, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
        [[0, 3, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
    ]
    cs["g0"] = [
        [[0, 0, 4], 3.0 * np.sqrt(1.0 / (np.pi)) / 2.0],
        [[4, 0, 0], 9.0 * np.sqrt(1.0 / (np.pi)) / 16.0],
        [[0, 4, 0], 9.0 * np.sqrt(1.0 / (np.pi)) / 16.0],
        [[2, 0, 2], -9.0 * np.sqrt(1.0 / np.pi) / 2.0],
        [[0, 2, 2], -9.0 * np.sqrt(1.0 / np.pi) / 2.0],
        [[2, 2, 0], 9.0 * np.sqrt(1.0 / np.pi) / 8.0],
    ]
    cs["g+1"] = [
        [[1, 0, 3], 3.0 * np.sqrt(5.0 / (2 * np.pi))],
        [[1, 2, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
        [[3, 0, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
    ]
    cs["g+2"] = [
        [[2, 0, 2], 18.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
        [[0, 2, 2], -18.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
        [[0, 4, 0], 3.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
        [[4, 0, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
    ]
    cs["g+3"] = [
        [[1, 2, 1], -9.0 * np.sqrt(35.0 / (2 * np.pi)) / 4.0],
        [[3, 0, 1], 0.75 * np.sqrt(35.0 / (2.0 * np.pi))],
    ]
    cs["g+4"] = [
        [[4, 0, 0], 3.0 * np.sqrt(35.0 / np.pi) / 16.0],
        [[2, 2, 0], -18.0 * np.sqrt(35.0 / np.pi) / 16.0],
        [[0, 4, 0], 3.0 * np.sqrt(35.0 / np.pi) / 16.0],
    ]
    ###############################################################################################################################
    ###############################################################################################################################
    Y1 = A2 * X / (A1 + A2)
    Y2 = -A1 * X / (A1 + A2)
    Z1 = cs[lm1]
    Z2 = cs[lm2]
    integral = np.sum(
        [
            Z1[it1][1] * Z2[it2][1] * KFunction(Y1, Y2, Z1[it1][0], Z2[it2][0], A1 + A2)
            for it1 in range(len(Z1))
            for it2 in range(len(Z2))
        ]
    )
    return integral


# -------------------------------------------------------------------------
def IInt(R1, A1, lm1, R2, A2, lm2):
    # computes the I integral using the J integral and the Gaussian prefactor
    # solid harmonics.
    # input:
    # R1:    (numpy.array)       position of nucleii 1
    # R2     (numpy.array)       position of nucleii 2
    # lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1)
    # lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2)
    # A1:    (positive real)     exponent gaussian of function 1
    # A2:    (positive real)     exponent of gaussian of function 2
    X = R1 - R2
    Jintegral = JInt(X, lm1, lm2, A1, A2)
    A12red = -A1 * A2 / (A1 + A2)
    Exponent = A12red * np.dot(X, X)
    gaussianPrefactor = np.exp(Exponent)
    integral = gaussianPrefactor * Jintegral
    return integral


# -------------------------------------------------------------------------
def getoverlap(R1, lm1, dalpha1, R2, lm2, dalpha2):
    # Compute overlap of two basis functions <phi_s1,n1,l1,m1(R_s1)|phi_s2,n2,l2,m2(R_s2)>
    # input:
    # R1:    (numpy.array)                                   position of nucleii 1
    # R2     (numpy.array)                                   position of nucleii 2
    # lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1)
    # dalpha1: (list of list)    specifies the first Gaussian type of wave function
    # lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2)
    # dalpha2: (list of list)    specifies the second Gaussian type of wave function
    overlap = 0.0
    overlap = np.sum(
        [
            dalpha1[it1][1]
            * dalpha2[it2][1]
            * IInt(R1, dalpha1[it1][0], lm1, R2, dalpha2[it2][0], lm2)
            for it1 in range(len(dalpha1))
            for it2 in range(len(dalpha2))
        ]
    )
    return overlap


# -------------------------------------------------------------------------
# For the Calculation of the TransitionDipolemoments


def OInt(X, lm1, lm2, A1, A2, k):
    # computes the J integral using the monomial decomposition of the
    # solid harmonics.
    # Input: X (numpy.array) of the difference vector R1-R2
    # A1: positive numerical
    # A2: positive numerical
    cs = {}

    cs["s"] = [[[0, 0, 0], 0.5 / np.sqrt(np.pi)]]

    cs["py"] = [[[0, 1, 0], np.sqrt(3.0 / (4.0 * np.pi))]]
    cs["pz"] = [[[0, 0, 1], np.sqrt(3.0 / (4.0 * np.pi))]]
    cs["px"] = [[[1, 0, 0], np.sqrt(3.0 / (4.0 * np.pi))]]

    cs["d-2"] = [[[1, 1, 0], 0.5 * np.sqrt(15.0 / np.pi)]]
    cs["d-1"] = [[[0, 1, 1], 0.5 * np.sqrt(15.0 / np.pi)]]
    cs["d0"] = [
        [[2, 0, 0], -0.25 * np.sqrt(5.0 / np.pi)],
        [[0, 2, 0], -0.25 * np.sqrt(5.0 / np.pi)],
        [[0, 0, 2], 0.5 * np.sqrt(5.0 / np.pi)],
    ]
    cs["d+1"] = [[[1, 0, 1], 0.5 * np.sqrt(15.0 / np.pi)]]
    cs["d+2"] = [
        [[2, 0, 0], 0.25 * np.sqrt(15.0 / np.pi)],
        [[0, 2, 0], -0.25 * np.sqrt(15.0 / np.pi)],
    ]

    cs["f-3"] = [
        [[2, 1, 0], 0.75 * np.sqrt(35.0 / 2.0 / np.pi)],
        [[0, 3, 0], -0.25 * np.sqrt(35.0 / 2.0 / np.pi)],
    ]
    cs["f-2"] = [[[1, 1, 1], 0.5 * np.sqrt(105.0 / np.pi)]]
    cs["f-1"] = [
        [[0, 1, 2], np.sqrt(21.0 / 2.0 / np.pi)],
        [[2, 1, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
        [[0, 3, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
    ]
    cs["f0"] = [
        [[0, 0, 3], 0.5 * np.sqrt(7.0 / np.pi)],
        [[2, 0, 1], -0.75 * np.sqrt(7 / np.pi)],
        [[0, 2, 1], -0.75 * np.sqrt(7 / np.pi)],
    ]
    cs["f+1"] = [
        [[1, 0, 2], np.sqrt(21.0 / 2.0 / np.pi)],
        [[1, 2, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
        [[3, 0, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
    ]
    cs["f+2"] = [
        [[2, 0, 1], 0.25 * np.sqrt(105.0 / np.pi)],
        [[0, 2, 1], -0.25 * np.sqrt(105.0 / np.pi)],
    ]
    cs["f+3"] = [
        [[3, 0, 0], 0.25 * np.sqrt(35.0 / 2.0 / np.pi)],
        [[1, 2, 0], -0.75 * np.sqrt(35.0 / 2.0 / np.pi)],
    ]

    cs["g-4"] = [
        [[3, 1, 0], 0.75 * np.sqrt(35.0 / np.pi)],
        [[1, 3, 0], -0.75 * np.sqrt(35.0 / np.pi)],
    ]
    cs["g-3"] = [
        [[2, 1, 1], 9.0 * np.sqrt(35.0 / (2 * np.pi)) / 4.0],
        [[0, 3, 1], -0.75 * np.sqrt(35.0 / (2.0 * np.pi))],
    ]
    cs["g-2"] = [
        [[1, 1, 2], 18.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
        [[3, 1, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
        [[1, 3, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
    ]
    cs["g-1"] = [
        [[0, 1, 3], 3.0 * np.sqrt(5.0 / (2 * np.pi))],
        [[2, 1, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
        [[0, 3, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
    ]
    cs["g0"] = [
        [[0, 0, 4], 3.0 * np.sqrt(1.0 / (np.pi)) / 2.0],
        [[4, 0, 0], 9.0 * np.sqrt(1.0 / (np.pi)) / 16.0],
        [[0, 4, 0], 9.0 * np.sqrt(1.0 / (np.pi)) / 16.0],
        [[2, 0, 2], -9.0 * np.sqrt(1.0 / np.pi) / 2.0],
        [[0, 2, 2], -9.0 * np.sqrt(1.0 / np.pi) / 2.0],
        [[2, 2, 0], 9.0 * np.sqrt(1.0 / np.pi) / 8.0],
    ]
    cs["g+1"] = [
        [[1, 0, 3], 3.0 * np.sqrt(5.0 / (2 * np.pi))],
        [[1, 2, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
        [[3, 0, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
    ]
    cs["g+2"] = [
        [[2, 0, 2], 18.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
        [[0, 2, 2], -18.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
        [[0, 4, 0], 3.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
        [[4, 0, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
    ]
    cs["g+3"] = [
        [[1, 2, 1], -9.0 * np.sqrt(35.0 / (2 * np.pi)) / 4.0],
        [[3, 0, 1], 0.75 * np.sqrt(35.0 / (2.0 * np.pi))],
    ]
    cs["g+4"] = [
        [[4, 0, 0], 3.0 * np.sqrt(35.0 / np.pi) / 16.0],
        [[2, 2, 0], -18.0 * np.sqrt(35.0 / np.pi) / 16.0],
        [[0, 4, 0], 3.0 * np.sqrt(35.0 / np.pi) / 16.0],
    ]

    Y1 = A2 * X / (A1 + A2)
    Y2 = -A1 * X / (A1 + A2)
    Z1 = cs[lm1]
    Z2 = cs[lm2]
    integral = 0.0
    if k == 0:
        for P1 in Z1:
            for P2 in Z2:
                c1 = P1[1]
                c2 = P2[1]
                is1 = P1[0]
                is2 = P2[0]
                integral += (
                    c1
                    * c2
                    * KFunction(Y1, Y2, (is1[0] + 1, is1[1], is1[2]), is2, A1 + A2)
                )
    elif k == 1:
        for P1 in Z1:
            for P2 in Z2:
                c1 = P1[1]
                c2 = P2[1]
                is1 = P1[0]
                is2 = P2[0]
                integral += (
                    c1
                    * c2
                    * KFunction(Y1, Y2, (is1[0], is1[1] + 1, is1[2]), is2, A1 + A2)
                )
    elif k == 2:
        for P1 in Z1:
            for P2 in Z2:
                c1 = P1[1]
                c2 = P2[1]
                is1 = P1[0]
                is2 = P2[0]
                integral += (
                    c1
                    * c2
                    * KFunction(Y1, Y2, (is1[0], is1[1], is1[2] + 1), is2, A1 + A2)
                )

    A12red = -A1 * A2 / (A1 + A2)
    Exponent = A12red * np.dot(X, X)
    gaussianPrefactor = np.exp(Exponent)
    integral = gaussianPrefactor * integral
    return integral


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def getContribution(R1, lm1, dalpha1, R2, lm2, dalpha2, k):
    # Compute overlap of two basis functions <phi_s1,n1,l1,m1(R_s1)|phi_s2,n2,l2,m2(R_s2)>
    # input:
    # R1:    (numpy.array)                                   position of nucleii 1
    # R2     (numpy.array)                                   position of nucleii 2
    # lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1)
    # dalpha1: (list of list)    specifies the first Gaussian type of wave function
    # lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2)
    # dalpha2: (list of list)    specifies the second Gaussian type of wave function
    overlap = 0.0
    for obj1 in dalpha1:
        for obj2 in dalpha2:
            d1 = obj1[1]
            alpha1 = obj1[0]
            d2 = obj2[1]
            alpha2 = obj2[0]
            overlap += d1 * d2 * OInt(R1 - R2, lm1, lm2, alpha1, alpha2, k)
    return overlap


# -------------------------------------------------------------------------
def getAngularMomentumString(l, m):
    ## Transforms the angular momentum notation (l,m) into the 's', 'py','pz','px' ect. notation
    ## input:   l                       angular momentum quantum number            (int)
    ##          m                       magnetic quantum number                    (int)
    ## output:  s                       the s- notation for the (l,m) pair         (string)
    if l == 0:
        s = "s"
    elif l == 1 and m == -1:
        s = "py"
    elif l == 1 and m == 0:
        s = "pz"
    elif l == 1 and m == 1:
        s = "px"
    elif l == 2:
        if m > 0:
            s = "d" + "+" + str(m)
        else:
            s = "d" + str(m)
    elif l == 3:
        if m > 0:
            s = "f" + "+" + str(m)
        else:
            s = "f" + str(m)
    elif l == 4:
        if m > 0:
            s = "g" + "+" + str(m)
        else:
            s = "g" + str(m)
    elif l == 5:
        if m > 0:
            s = "h" + "+" + str(m)
        else:
            s = "h" + str(m)
    else:
        print("Higher order not yet implemented")
    return s


# -------------------------------------------------------------------------
def getNormalizationfactor(alpha, l):
    ## Transformationfactor between the normalized contracted cartesian Basis set from the data directory,
    ## and the not normalized Basis set used in the QS routines of cp2k.
    ## This means the contraction coefficients c_dd from the data directory are connected with
    ## those used in the QS routines c_QS via c_QS(alpha,l)=Output(alpha,l)*c_dd(alpha,l),
    ## where Output is the output of this function.
    ## input:   alpha                       the exponent of the Gaussian            (float)
    ##          l                           the angular momentum quantum number     (int)
    ## output:  The transformation factor
    return alpha ** (0.5 * l + 0.75) * 2 ** (l) * (2.0 / np.pi) ** (0.75)


# -------------------------------------------------------------------------


def getBasisSetName(path, cp2kpath=pathtocp2k):
    ## Reads in from the .inp file in path the Basis sets used. Parses the corresponding
    ## data from the cp2kpath/data/Basis_Set file and returns this parsed data as a list.
    ## Each element in this list is a string of the corresponding line in the Basis set file
    ## input:   path                path to the folder of the .inp file         (string)
    ## (opt.)   cp2kpath            path to the cp2k folder                     (string)
    ## output:  BasisInfoReadin                                                 (list of strings)

    # open the .inp file
    inpfile = [f for f in os.listdir(path) if f.endswith(".inp")]
    if len(inpfile) != 1:
        raise ValueError(
            "InputError: There should be only one inp file in the current directory"
        )
    atoms = []
    BasisSetNames = []
    BasisSetFileName = "empty"
    BasisSetNameFlag = False
    with open(path + "/" + inpfile[0], "r") as g:
        lines = g.readlines()
        for line in lines:
            if len(line.split()) > 0:
                if line.split()[0] == "BASIS_SET_FILE_NAME":
                    BasisSetFileName = line.split()[1]
                if line.split()[0] == "&KIND":
                    atoms.append(line.split()[1])
                    BasisSetNameFlag = True
                if line.split()[0] == "END" and line.split()[1] == "&KIND":
                    BasisSetNameFlag = False
                if BasisSetNameFlag and line.split()[0] == "BASIS_SET":
                    BasisSetNames.append(line.split()[1])
    atomStrings = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
    ]
    print("Used Basis:", BasisSetFileName)
    BasisInfoReadin = []
    ReadinFlag = False
    for it in range(len(atoms)):
        with open(cp2kpath + "/data/" + BasisSetFileName, "r") as g:
            for l in g:
                if len(l.split()) >= 1:
                    numericflag = l.split()[0][0].isnumeric()
                if len(l.split()) == 2 and not numericflag:
                    if l.split()[0] == atoms[it] and l.split()[1] == BasisSetNames[it]:
                        ReadinFlag = True
                    if (l.split()[0] in atomStrings and l.split()[0] not in atoms) or (
                        l.split()[1] != BasisSetNames[it]
                    ):
                        ReadinFlag = False
                elif len(l.split()) > 2 and not numericflag:
                    if (
                        l.split()[0] == atoms[it] and l.split()[1] == BasisSetNames[it]
                    ) or (
                        l.split()[0] == atoms[it] and l.split()[2] == BasisSetNames[it]
                    ):
                        ReadinFlag = True
                    elif bool(
                        l.split()[0] in atomStrings and l.split()[0] not in atoms
                    ) or (
                        bool(l.split()[1] != BasisSetNames[it])
                        or bool(l.split()[2] != BasisSetNames[it])
                    ):
                        ReadinFlag = False
                if ReadinFlag:
                    BasisInfoReadin.append(l)
    for it in range(len(BasisInfoReadin)):
        item = BasisInfoReadin[it]
        item = item[:-1]
        BasisInfoReadin[it] = item
    return BasisInfoReadin


# -------------------------------------------------------------------------
def getBasis(filename):
    ##Constructs the (non-orthorgonal) Basis used in the CP2K calculation
    ## input:
    ## (opt.)   filename            path to the calculation folder                   (string)
    ## output:  Basis               dic. of lists of sublists. The keys of the dic. are
    ##                              the atomic symbols.
    ##                              list contains sublist, where each Basisfunction of the
    ##                              considered atom corresponds the one sublist.
    ##                              sublist[0] contains the set index as a string.
    ##                              sublist[1] contains the shell index as a string
    ##                              sublist[2] contains the angular momentum label
    ##                              as a string (e.g. shellindex py ect.)
    ##                              sublist[3:] are lists with two elements.
    ##                              The first corresponds the the exponent of the Gaussian
    ##                              The second one corresponds to the contraction coefficient
    BasisInfoReadin = getBasisSetName(filename)
    atoms = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
    ]
    BasisSet = {}
    atom = "NotDefined"
    Atombasis = []
    newAtomSetflag = False
    newSetflag = False
    readinBasisfunctionsflag = False
    setcounter = 1
    linecounter = 0
    NumberofExponents = 0
    numberofBasissets = 0
    for line in BasisInfoReadin:
        splitedline = line.split()
        if len(splitedline) > 0:
            firstcaracter = splitedline[0]
            if readinBasisfunctionsflag:
                if linecounter == NumberofExponents + 1:
                    readinBasisfunctionsflag = False
                    newSetflag = True
                    setcounter += 1
                else:
                    exponent = float(splitedline[0])
                    coefficientiter = 1
                    it = 0
                    for it1 in range(len(ls)):
                        l = ls[it1]
                        for it2 in range(NumofangularmomentumBasisfunctions[it1]):
                            coefficient = float(splitedline[coefficientiter])
                            for m in range(-l, l + 1):
                                Basisfunctions[it].append(
                                    [
                                        exponent,
                                        getNormalizationfactor(exponent, l)
                                        * coefficient,
                                    ]
                                )  #
                                it += 1
                            coefficientiter += 1
                    if linecounter == NumberofExponents:
                        Atombasis.append(Basisfunctions)
                    linecounter += 1
            if newSetflag and setcounter <= numberofBasissets:
                minprincipalQuantumnumber = int(splitedline[0])
                lmin = int(splitedline[1])
                lmax = int(splitedline[2])
                ls = np.array([lmin + it for it in range(0, lmax - lmin + 1)])
                NumberofExponents = int(splitedline[3])
                NumofangularmomentumBasisfunctions = []
                for split in splitedline[4:]:
                    NumofangularmomentumBasisfunctions.append(int(split))
                if (lmax - lmin + 1) != len(NumofangularmomentumBasisfunctions):
                    ValueError("Number of Basis functions does not fit!")
                NumofangularmomentumBasisfunctions = np.array(
                    NumofangularmomentumBasisfunctions
                )
                readinBasisfunctionsflag = True
                newSetflag = False
                shell = 0
                Basisfunctions = []
                for it1 in range(len(ls)):
                    l = ls[it1]
                    for it2 in range(NumofangularmomentumBasisfunctions[it1]):
                        shell += 1
                        for m in range(-l, l + 1):
                            Basisfunctions.append(
                                [
                                    str(setcounter),
                                    str(shell),
                                    getAngularMomentumString(l, m),
                                ]
                            )
                linecounter = 1
            if newAtomSetflag:
                numberofBasissets = int(firstcaracter)
                setcounter = 1
                newAtomSetflag = False
                newSetflag = True
            if setcounter == numberofBasissets and linecounter == NumberofExponents + 1:
                Basis = []
                for it1 in range(len(Atombasis)):
                    for it2 in range(len(Atombasis[it1])):
                        Basis.append(Atombasis[it1][it2])
                BasisSet[atom] = Basis
            if firstcaracter in atoms:
                Atombasis = []
                atom = firstcaracter
                newAtomSetflag = True
    # Normalize the Basis
    for key in BasisSet.keys():
        for it in range(len(BasisSet[key])):
            lm1 = BasisSet[key][it][2][:]
            lm2 = lm1
            dalpha1 = BasisSet[key][it][3:]
            dalpha2 = dalpha1
            # C
            # Normalize the Basis
            R1 = np.array([0.0, 0.0, 0.0])
            R2 = np.array([0.0, 0.0, 0.0])
            normfactor = 1.0 / np.sqrt(getoverlap(R1, lm1, dalpha1, R2, lm2, dalpha2))
            for it2 in range(len(BasisSet[key][it]) - 3):
                BasisSet[key][it][it2 + 3][1] *= normfactor
    return BasisSet


# -------------------------------------------------------------------------
def getTransformationmatrix(
    Atoms1: list[tuple[int, str, float, float, float]],
    Atoms2: list[tuple[int, str, float, float, float]],
    Basis: dict[str, list[tuple[str, str, str, tuple[float, float], ...]]],
    cell_vectors: list[float] = [0.0, 0.0, 0.0],
):
    ##Compute the overlap & transformation matrix of the Basis functions with respect to the conventional basis ordering
    ##input: Atoms1              atoms of the first index
    ##                           list of sublists.
    ##                           Each of the sublists has five elements.
    ##                           Sublist[0] contains the atomorder as a int.
    ##                           Sublist[1] contains the symbol of the atom.
    ##                           Sublist[2:] containst the x y z coordinates.
    ##                                       unit: Angstroem
    ##
    ##       Atoms2              Atoms of the second index
    ##
    ##       Basis               dic. of lists of sublists. The keys of the dic. are
    ##                           the atomic symbols.
    ##                           list contains sublist, where each Basisfunction of the
    ##                           considered atom corresponds the one sublist.
    ##                           sublist[0] contains the set index as a string.
    ##                           sublist[1] contains the shell index as a string
    ##                           sublist[2] contains the angular momentum label
    ##                           as a string (e.g. shellindex py ect.)
    ##                           sublist[3:] are lists with two elements.
    ##                           The first corresponds to the exponent of the Gaussian
    ##                           The second one corresponds to the contraction coefficient
    ##
    ##
    ##     cellvectors (opt.)    different cell vectors to take into account in the calculation the default implements open boundary conditions
    ##output:   Overlapmatrix    The Transformation matrix as a numpy array

    # Initialize the python lists for Basis Set 1
    atoms_set1: list[str] = []
    positions_set1: list[float] = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms1)):
        Atom_type1 = Atoms1[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = (
                np.array(Atoms1[itAtom1][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            # R1: (3,)
            positions_set1.extend(R1)

            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = (
        alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1
    )

    # Initialize the python lists for Basis Set 2
    atoms_set2: list[str] = []
    positions_set2: list[float] = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms2)):
        Atom_type2 = Atoms2[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = (
                np.array(Atoms2[itAtom2][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            positions_set2.extend(R2)

            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = (
        alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1
    )

    # Call the C++ function
    OLP_array = get_T_Matrix(
        atoms_set1=atoms_set1,
        positions_set1=positions_set1,
        alphas_set1=alphas_set1,
        alphasLengths_set1=alphas_lengths_set1,
        contr_coef_set1=contr_coef_set1,
        contr_coefLengths_set1=contr_coef_lengths_set1,
        lms_set1=lms_set1,
        atoms_set2=atoms_set2,
        positions_set2=positions_set2,
        alphas_set2=alphas_set2,
        alphasLengths_set2=alphas_lengths_set2,
        contr_coef_set2=contr_coef_set2,
        contr_coefLengths_set2=contr_coef_lengths_set2,
        lms_set2=lms_set2,
        cell_vectors=cell_vectors,
    )

    OLP = np.array(OLP_array).reshape((len(atoms_set1), len(atoms_set2)))

    return OLP


def WFNonxyzGrid(
    grid1: np.ndarray,
    grid2: np.ndarray,
    grid3: np.ndarray,
    Coefficients: list[float],
    Atoms: list[tuple[int, str, float, float, float]],
    Basis: dict[str, list[tuple[str, str, str, tuple[float, float], ...]]],
    cell_vectors: list[float] = [0.0, 0.0, 0.0],
):
    ## TODO: I don't think this is the correct documentation here?
    ##Compute the overlap & transformation matrix of the Basis functions with respect to the conventional basis ordering
    ##input: Atoms              atoms of the first index
    ##                           list of sublists.
    ##                           Each of the sublists has five elements.
    ##                           Sublist[0] contains the atomorder as a int.
    ##                           Sublist[1] contains the symbol of the atom.
    ##                           Sublist[2:] containst the x y z coordinates.
    ##                                       unit: Angstroem
    ##
    ##
    ##       Basis               dic. of lists of sublists. The keys of the dic. are
    ##                           the atomic symbols.
    ##                           list contains sublist, where each Basisfunction of the
    ##                           considered atom corresponds the one sublist.
    ##                           sublist[0] contains the set index as a string.
    ##                           sublist[1] contains the shell index as a string
    ##                           sublist[2] contains the angular momentum label
    ##                           as a string (e.g. shellindex py ect.)
    ##                           sublist[3:] are lists with two elements.
    ##                           The first corresponds the the exponent of the Gaussian
    ##                           The second one corresponds to the contraction coefficient
    ##
    ##
    ##     cellvectors (opt.)    different cell vectors to take into account in the calculation the default implements open boundary conditions
    ##output:   Overlapmatrix    The Transformation matrix as a numpy array

    # Proposed new docstring
    """
    Compute wavefunction values on a 3D grid.

    Args:
        grid1, grid2, grid3: 1D numpy arrays defining the 3D grid points.
        coefficients: list of wavefunction coefficients.
        atoms: list of (index, symbol, x, y, z).
        basis: dictionary defining basis functions per atom type.
        cell_vectors: lattice vectors (default zeros for non-periodic systems).

    Returns:
        A 3D numpy array with shape (len(grid1), len(grid2), len(grid3)).
    """

    # Make the grid
    xyz_grid = (
        np.array(np.meshgrid(grid1, grid2, grid3, indexing="ij")).flatten("F").tolist()
        #np.array(np.meshgrid(grid1, grid2, grid3, indexing="ij")).ravel(order="C").tolist()
    )

    # Initialize the python lists for Basis Set
    atoms_set: list[str] = []
    positions_set: list[float] = []
    alphas_lengths_set = []
    alphas_set = []
    contr_coef_set = []
    lms_set: list[str] = []

    # Create Python lists for input (Set 1)
    for itAtom in range(len(Atoms)):
        Atom_type = Atoms[itAtom][1]
        B = Basis[Atom_type]
        for itBasis in range(len(Basis[Atom_type])):
            atoms_set.append(Atom_type)
            R = (
                np.array(Atoms[itAtom][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            # R: (3,)
            positions_set.extend(R)

            state = B[itBasis]
            dalpha = state[3:]
            alphas_lengths_set.append(len(dalpha))
            for it2 in range(len(dalpha)):
                alphas_set.append(dalpha[it2][0])
                contr_coef_set.append(dalpha[it2][1])
            lm = state[2][:]
            lms_set.append(lm)

    contr_coef_lengths_set = (
        alphas_lengths_set  # Lengths of contr_coef for each basis function in Set 1
    )

    # Call the C++ function
    WFN_On_Grid_array = get_WFN_On_Grid(
        xyzgrid=xyz_grid,
        WFNcoefficients=Coefficients,
        atoms_set=atoms_set,
        positions_set=positions_set,
        alphas_set=alphas_set,
        alphasLengths_set=alphas_lengths_set,
        contr_coef_set=contr_coef_set,
        contr_coefLengths_set=contr_coef_lengths_set,
        lms_set=lms_set,
        cell_vectors=cell_vectors,
    )

    # reshape the output array
    WFN_on_Grid = np.array(WFN_On_Grid_array, dtype=float)
    WFN_on_Grid = WFN_on_Grid.reshape((len(grid1), len(grid2), len(grid3)), order="F")

    return WFN_on_Grid


def LocalPotentialonxyzGrid(
    gridpoints,
    MatrixElements,
    Atoms,
    Basis,
    cell_vectors=[0.0, 0.0, 0.0],
):
    ## TODO: I don't think this is the correct documentation here?
    ##Compute the overlap & transformation matrix of the Basis functions with respect to the conventional basis ordering
    ##input: Atoms              atoms of the first index
    ##                           list of sublists.
    ##                           Each of the sublists has five elements.
    ##                           Sublist[0] contains the atomorder as a int.
    ##                           Sublist[1] contains the symbol of the atom.
    ##                           Sublist[2:] containst the x y z coordinates.
    ##                                       unit: Angstroem
    ##
    ##
    ##       Basis               dic. of lists of sublists. The keys of the dic. are
    ##                           the atomic symbols.
    ##                           list contains sublist, where each Basisfunction of the
    ##                           considered atom corresponds the one sublist.
    ##                           sublist[0] contains the set index as a string.
    ##                           sublist[1] contains the shell index as a string
    ##                           sublist[2] contains the angular momentum label
    ##                           as a string (e.g. shellindex py ect.)
    ##                           sublist[3:] are lists with two elements.
    ##                           The first corresponds the the exponent of the Gaussian
    ##                           The second one corresponds to the contraction coefficient
    ##
    ##
    ##     cellvectors (opt.)    different cell vectors to take into account in the calculation the default implements open boundary conditions
    ##output:   Overlapmatrix    The Transformation matrix as a numpy array

    # Make the grid
    xyz_grid = gridpoints

    # Initialize the python lists for Basis Set
    atoms_set = []
    positions_set = []
    alphas_lengths_set = []
    alphas_set = []
    contr_coef_set = []
    lms_set = []

    # Create Python lists for input (Set 1)
    for itAtom in range(len(Atoms)):
        Atom_type = Atoms[itAtom][1]
        B = Basis[Atom_type]
        for itBasis in range(len(Basis[Atom_type])):
            atoms_set.append(Atom_type)
            R = (
                np.array(Atoms[itAtom][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            positions_set.extend(R)

            state = B[itBasis]
            dalpha = state[3:]
            alphas_lengths_set.append(len(dalpha))
            for it2 in range(len(dalpha)):
                alphas_set.append(dalpha[it2][0])
                contr_coef_set.append(dalpha[it2][1])
            lm = state[2][:]
            lms_set.append(lm)

    contr_coef_lengths_set = (
        alphas_lengths_set  # Lengths of contr_coef for each basis function in Set 1
    )

    MatrixElementsList = []
    for it1 in range(np.shape(MatrixElements)[0]):
        for it2 in range(np.shape(MatrixElements)[1]):
            MatrixElementsList.append(MatrixElements[it1][it2])

    # Call the C++ function
    LocalPotentialonGrid = get_Local_Potential_On_Grid(
        xyzgrid=xyz_grid,
        MatrixElements=MatrixElementsList,
        atoms_set=atoms_set,
        positions_set=positions_set,
        alphas_set=alphas_set,
        alphasLengths_set=alphas_lengths_set,
        contr_coef_set=contr_coef_set,
        contr_coefLengths_set=contr_coef_lengths_set,
        lms_set=lms_set,
        cell_vectors=cell_vectors,
    )

    LocalPotentialonGrid = np.array(LocalPotentialonGrid, dtype=float)

    return LocalPotentialonGrid


def get_Position_Operators(
    Atoms: list[tuple[int, str, float, float, float]],
    Basis: dict[str, list[tuple[str, str, str, tuple[float, float], ...]]],
    cell_vectors: list[float] = [0.0, 0.0, 0.0],
):
    """
    Constructs the position operators (x, y, z) in matrix form for a given
    set of atoms and basis functions.

    Parameters
    ----------
    Atoms : list
        A list of atoms, where each atom is represented as a list containing:
        - An identifier (=atom index)
        - Atom type (key for looking up basis functions)
        - Cartesian coordinates [x, y, z] in Angstrom units

    Basis : dict
        A dictionary containing basis sets for each atom type.
        Example:
            Basis = {
                'H': [['1', '1', 's', [1, 2.5264751109842596]]],
                'C': [['1', '1', 's', [2, 2.5264751109842596/np.sqrt(0.35355339)]]],
                ...
            }

    cellvectors : list, optional
        A list of three values representing the cell vectors of the simulation cell
        (for periodic systems). Defaults to [0.0, 0.0, 0.0].

    Returns
    -------
    x_op : numpy.ndarray
        Matrix representation of the x position operator in the atomic basis.

    y_op : numpy.ndarray
        Matrix representation of the y position operator in the atomic basis.

    z_op : numpy.ndarray
        Matrix representation of the z position operator in the atomic basis.

    Notes
    -----
    - Atom positions are converted from Angstroms to atomic units using a conversion factor.
    - Position-dependent transformation matrices are obtained via `getTransformationmatrix()`:
        - modus=0: Overlap-like matrix (OLM)
        - modus=1: x operator base matrix
        - modus=2: y operator base matrix
        - modus=3: z operator base matrix
    - The final position operators are computed as:
        - x_op = x_op_1 + Rx * OLM
        - y_op = y_op_1 + Ry * OLM
        - z_op = z_op_1 + Rz * OLM
    - The function currently returns after processing the first atom’s basis functions.
      This behavior should be reviewed if the intention was to process all atoms.

    Example
    -------
    Atoms = [
        [0, 'H', 0.0, 0.0, 0.0],
        [1, 'H', 0.74, 0.0, 0.0]
    ]
    Basis = {
        'H': ['1s', '2s']
    }
    x_op, y_op, z_op = getPositionOperator(Atoms, Basis)
    """

    # Initialize the python lists for Basis Set 1
    atoms_set1: list[str] = []
    positions_set1: list[float] = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms)):
        Atom_type1 = Atoms[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = (
                np.array(Atoms[itAtom1][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            positions_set1.extend(R1)

            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = (
        alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1
    )

    # Initialize the python lists for Basis Set 2
    atoms_set2: list[str] = []
    positions_set2: list[float] = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms)):
        Atom_type2 = Atoms[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = (
                np.array(Atoms[itAtom2][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            positions_set2.extend(R2)

            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = (
        alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1
    )

    # Call the C++ function
    # TODO: Nomenclature: get_Position_Operators returns an operator and not an overlap?
    # So the name is a bit misleading...
    # TODO: Reduce code duplication by iteration over directions
    OLP_array = get_position_operators(
        atoms_set1=atoms_set1,
        positions_set1=positions_set1,
        alphas_set1=alphas_set1,
        alphasLengths_set1=alphas_lengths_set1,
        contr_coef_set1=contr_coef_set1,
        contr_coefLengths_set1=contr_coef_lengths_set1,
        lms_set1=lms_set1,
        atoms_set2=atoms_set2,
        positions_set2=positions_set2,
        alphas_set2=alphas_set2,
        alphasLengths_set2=alphas_lengths_set2,
        contr_coef_set2=contr_coef_set2,
        contr_coefLengths_set2=contr_coef_lengths_set2,
        lms_set2=lms_set2,
        cell_vectors=cell_vectors,
        direction=1,
    )

    x_operator = np.array(OLP_array).reshape((len(atoms_set1), len(atoms_set2)))

    # Call the C++ function
    OLP_array = get_position_operators(
        atoms_set1=atoms_set1,
        positions_set1=positions_set1,
        alphas_set1=alphas_set1,
        alphasLengths_set1=alphas_lengths_set1,
        contr_coef_set1=contr_coef_set1,
        contr_coefLengths_set1=contr_coef_lengths_set1,
        lms_set1=lms_set1,
        atoms_set2=atoms_set2,
        positions_set2=positions_set2,
        alphas_set2=alphas_set2,
        alphasLengths_set2=alphas_lengths_set2,
        contr_coef_set2=contr_coef_set2,
        contr_coefLengths_set2=contr_coef_lengths_set2,
        lms_set2=lms_set2,
        cell_vectors=cell_vectors,
        direction=2,
    )
    y_operator = np.array(OLP_array).reshape((len(atoms_set1), len(atoms_set2)))

    # Call the C++ function
    OLP_array = get_position_operators(
        atoms_set1=atoms_set1,
        positions_set1=positions_set1,
        alphas_set1=alphas_set1,
        alphasLengths_set1=alphas_lengths_set1,
        contr_coef_set1=contr_coef_set1,
        contr_coefLengths_set1=contr_coef_lengths_set1,
        lms_set1=lms_set1,
        atoms_set2=atoms_set2,
        positions_set2=positions_set2,
        alphas_set2=alphas_set2,
        alphasLengths_set2=alphas_lengths_set2,
        contr_coef_set2=contr_coef_set2,
        contr_coefLengths_set2=contr_coef_lengths_set2,
        lms_set2=lms_set2,
        cell_vectors=cell_vectors,
        direction=3,
    )

    z_operator = np.array(OLP_array).reshape((len(atoms_set1), len(atoms_set2)))

    return (
        x_operator / ConversionFactors["A->a.u."],
        y_operator / ConversionFactors["A->a.u."],
        z_operator / ConversionFactors["A->a.u."],
    )


def get_momentum_operators(
    Atoms: list[tuple[int, str, float, float, float]],
    Basis: dict[str, list[tuple[str, str, str, tuple[float, float], ...]]],
    cell_vectors=[0.0, 0.0, 0.0]
):

    # Initialize the python lists for Basis Set 1
    atoms_set1: list[str] = []
    positions_set1: list[float] = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # TODO: Create a helper function to reduce code duplication, improve readability and reduce errors
    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms)):
        Atom_type1 = Atoms[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = (
                np.array(Atoms[itAtom1][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            positions_set1.extend(R1)

            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = (
        alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1
    )

    # Initialize the python lists for Basis Set 2
    atoms_set2: list[str] = []
    positions_set2: list[float] = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms)):
        Atom_type2 = Atoms[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = (
                np.array(Atoms[itAtom2][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            positions_set2.extend(R2)

            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = (
        alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1
    )

    # Call the C++ function
    ## TODO: Nomenclature - get_Momentum_Operators returns an operator and not an overlap?
    ## TODO: Iterate over directions to reduce code duplication
    OLP_array = get_Momentum_Operators(
        atoms_set1=atoms_set1,
        positions_set1=positions_set1,
        alphas_set1=alphas_set1,
        alphasLengths_set1=alphas_lengths_set1,
        contr_coef_set1=contr_coef_set1,
        contr_coefLengths_set1=contr_coef_lengths_set1,
        lms_set1=lms_set1,
        atoms_set2=atoms_set2,
        positions_set2=positions_set2,
        alphas_set2=alphas_set2,
        alphasLengths_set2=alphas_lengths_set2,
        contr_coef_set2=contr_coef_set2,
        contr_coefLengths_set2=contr_coef_lengths_set2,
        lms_set2=lms_set2,
        cell_vectors=cell_vectors,
        direction=1,
    )

    p_x = np.array(OLP_array).reshape((len(atoms_set1), len(atoms_set2)))

    # Call the C++ function
    OLP_array = get_Momentum_Operators(
        atoms_set1=atoms_set1,
        positions_set1=positions_set1,
        alphas_set1=alphas_set1,
        alphasLengths_set1=alphas_lengths_set1,
        contr_coef_set1=contr_coef_set1,
        contr_coefLengths_set1=contr_coef_lengths_set1,
        lms_set1=lms_set1,
        atoms_set2=atoms_set2,
        positions_set2=positions_set2,
        alphas_set2=alphas_set2,
        alphasLengths_set2=alphas_lengths_set2,
        contr_coef_set2=contr_coef_set2,
        contr_coefLengths_set2=contr_coef_lengths_set2,
        lms_set2=lms_set2,
        cell_vectors=cell_vectors,
        direction=2,
    )

    p_y = np.array(OLP_array).reshape((len(atoms_set1), len(atoms_set2)))

    # Call the C++ function
    OLP_array = get_Momentum_Operators(
        atoms_set1=atoms_set1,
        positions_set1=positions_set1,
        alphas_set1=alphas_set1,
        alphasLengths_set1=alphas_lengths_set1,
        contr_coef_set1=contr_coef_set1,
        contr_coefLengths_set1=contr_coef_lengths_set1,
        lms_set1=lms_set1,
        atoms_set2=atoms_set2,
        positions_set2=positions_set2,
        alphas_set2=alphas_set2,
        alphasLengths_set2=alphas_lengths_set2,
        contr_coef_set2=contr_coef_set2,
        contr_coefLengths_set2=contr_coef_lengths_set2,
        lms_set2=lms_set2,
        cell_vectors=cell_vectors,
        direction=3,
    )

    p_z = np.array(OLP_array).reshape((len(atoms_set1), len(atoms_set2)))

    return -1.0j * p_x, -1.0j * p_y, -1.0j * p_z


# Define a struct matching std::complex<double>
#class ComplexDouble(Structure):
#    _fields_ = [("real", c_double), ("imag", c_double)]

#    def __repr__(self):
#        return f"ComplexDouble(real={self.real}, imag={self.imag})"


def get_phase_operators(
    Atoms: list[tuple[int, str, float, float, float]],
    Basis: dict[str, list[tuple[str, str, str, tuple[float, float], ...]]],
    q_vector: list[float] = [0.0, 0.0, 0.0],
    cell_vectors=[0.0, 0.0, 0.0],
):
    ## TODO: Docstring is missing
    # Initialize the python lists for Basis Set 1
    atoms_set1: list[str] = []
    positions_set1: list[float] = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms)):
        Atom_type1 = Atoms[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = (
                np.array(Atoms[itAtom1][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            positions_set1.extend(R1)

            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = (
        alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1
    )

    # Initialize the python lists for Basis Set 2
    atoms_set2: list[str] = []
    positions_set2: list[float] = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms)):
        Atom_type2 = Atoms[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = (
                np.array(Atoms[itAtom2][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            positions_set2.extend(R2)

            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = (
        alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1
    )

    # Call the C++ function
    ## TODO: Nomenclature - get_Phase_Operators returns an operator and not an overlap?
    OLP_array = get_Phase_Operators(
        atoms_set1=atoms_set1,
        positions_set1=positions_set1,
        alphas_set1=alphas_set1,
        alphasLengths_set1=alphas_lengths_set1,
        contr_coef_set1=contr_coef_set1,
        contr_coefLengths_set1=contr_coef_lengths_set1,
        lms_set1=lms_set1,
        atoms_set2=atoms_set2,
        positions_set2=positions_set2,
        alphas_set2=alphas_set2,
        alphasLengths_set2=alphas_lengths_set2,
        contr_coef_set2=contr_coef_set2,
        contr_coefLengths_set2=contr_coef_lengths_set2,
        lms_set2=lms_set2,
        cell_vectors=cell_vectors,
        q=q_vector,
    )

    phi_q = np.array(OLP_array, dtype=np.complex128).reshape((len(atoms_set1), len(atoms_set2)))

    return phi_q
