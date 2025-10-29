"""
Atomic Basis cpp extension module
"""
from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['get_Local_Potential_On_Grid', 'get_Momentum_Operators', 'get_Phase_Operators', 'get_T_Matrix', 'get_WFN_On_Grid', 'get_position_operators']
def get_Local_Potential_On_Grid(xyzgrid: collections.abc.Sequence[typing.SupportsFloat], MatrixElements: collections.abc.Sequence[typing.SupportsFloat], atoms_set: collections.abc.Sequence[str], positions_set: collections.abc.Sequence[typing.SupportsFloat], alphas_set: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set: collections.abc.Sequence[typing.SupportsInt], contr_coef_set: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set: collections.abc.Sequence[typing.SupportsInt], lms_set: collections.abc.Sequence[str], cell_vectors: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
    """
    Get local potential on grid
    """
def get_Momentum_Operators(atoms_set1: collections.abc.Sequence[str], positions_set1: collections.abc.Sequence[typing.SupportsFloat], alphas_set1: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set1: collections.abc.Sequence[typing.SupportsInt], contr_coef_set1: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set1: collections.abc.Sequence[typing.SupportsInt], lms_set1: collections.abc.Sequence[str], atoms_set2: collections.abc.Sequence[str], positions_set2: collections.abc.Sequence[typing.SupportsFloat], alphas_set2: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set2: collections.abc.Sequence[typing.SupportsInt], contr_coef_set2: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set2: collections.abc.Sequence[typing.SupportsInt], lms_set2: collections.abc.Sequence[str], cell_vectors: collections.abc.Sequence[typing.SupportsFloat], direction: typing.SupportsInt) -> list[float]:
    """
    Get momentum operator matrix elements
    """
def get_Phase_Operators(atoms_set1: collections.abc.Sequence[str], positions_set1: collections.abc.Sequence[typing.SupportsFloat], alphas_set1: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set1: collections.abc.Sequence[typing.SupportsInt], contr_coef_set1: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set1: collections.abc.Sequence[typing.SupportsInt], lms_set1: collections.abc.Sequence[str], atoms_set2: collections.abc.Sequence[str], positions_set2: collections.abc.Sequence[typing.SupportsFloat], alphas_set2: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set2: collections.abc.Sequence[typing.SupportsInt], contr_coef_set2: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set2: collections.abc.Sequence[typing.SupportsInt], lms_set2: collections.abc.Sequence[str], cell_vectors: collections.abc.Sequence[typing.SupportsFloat], q: collections.abc.Sequence[typing.SupportsFloat]) -> list[complex]:
    """
    Get phase operator matrix elements
    """
def get_T_Matrix(atoms_set1: collections.abc.Sequence[str], positions_set1: collections.abc.Sequence[typing.SupportsFloat], alphas_set1: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set1: collections.abc.Sequence[typing.SupportsInt], contr_coef_set1: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set1: collections.abc.Sequence[typing.SupportsInt], lms_set1: collections.abc.Sequence[str], atoms_set2: collections.abc.Sequence[str], positions_set2: collections.abc.Sequence[typing.SupportsFloat], alphas_set2: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set2: collections.abc.Sequence[typing.SupportsInt], contr_coef_set2: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set2: collections.abc.Sequence[typing.SupportsInt], lms_set2: collections.abc.Sequence[str], cell_vectors: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
    """
    Get transformation matrix
    """
def get_WFN_On_Grid(xyzgrid: collections.abc.Sequence[typing.SupportsFloat], WFNcoefficients: collections.abc.Sequence[typing.SupportsFloat], atoms_set: collections.abc.Sequence[str], positions_set: collections.abc.Sequence[typing.SupportsFloat], alphas_set: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set: collections.abc.Sequence[typing.SupportsInt], contr_coef_set: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set: collections.abc.Sequence[typing.SupportsInt], lms_set: collections.abc.Sequence[str], cell_vectors: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
    """
    Get wavefunction on grid
    """
def get_position_operators(atoms_set1: collections.abc.Sequence[str], positions_set1: collections.abc.Sequence[typing.SupportsFloat], alphas_set1: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set1: collections.abc.Sequence[typing.SupportsInt], contr_coef_set1: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set1: collections.abc.Sequence[typing.SupportsInt], lms_set1: collections.abc.Sequence[str], atoms_set2: collections.abc.Sequence[str], positions_set2: collections.abc.Sequence[typing.SupportsFloat], alphas_set2: collections.abc.Sequence[typing.SupportsFloat], alphasLengths_set2: collections.abc.Sequence[typing.SupportsInt], contr_coef_set2: collections.abc.Sequence[typing.SupportsFloat], contr_coefLengths_set2: collections.abc.Sequence[typing.SupportsInt], lms_set2: collections.abc.Sequence[str], cell_vectors: collections.abc.Sequence[typing.SupportsFloat], direction: typing.SupportsInt) -> list[float]:
    """
    Get position operator matrix elements
    """
