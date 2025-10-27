"""
Atomic Basis cpp extension module
"""
from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['get_Local_Potential_On_Grid', 'get_Momentum_Operators', 'get_Phase_Operators', 'get_Position_Operators', 'get_T_Matrix', 'get_WFN_On_Grid']
def get_Local_Potential_On_Grid(arg0: collections.abc.Sequence[typing.SupportsFloat], arg1: collections.abc.Sequence[typing.SupportsFloat], arg2: collections.abc.Sequence[str], arg3: collections.abc.Sequence[typing.SupportsFloat], arg4: collections.abc.Sequence[typing.SupportsFloat], arg5: collections.abc.Sequence[typing.SupportsInt], arg6: collections.abc.Sequence[typing.SupportsFloat], arg7: collections.abc.Sequence[typing.SupportsInt], arg8: collections.abc.Sequence[str], arg9: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
    """
    Get local potential on grid
    """
def get_Momentum_Operators(arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[typing.SupportsFloat], arg2: collections.abc.Sequence[typing.SupportsFloat], arg3: collections.abc.Sequence[typing.SupportsInt], arg4: collections.abc.Sequence[typing.SupportsFloat], arg5: collections.abc.Sequence[typing.SupportsInt], arg6: collections.abc.Sequence[str], arg7: collections.abc.Sequence[str], arg8: collections.abc.Sequence[typing.SupportsFloat], arg9: collections.abc.Sequence[typing.SupportsFloat], arg10: collections.abc.Sequence[typing.SupportsInt], arg11: collections.abc.Sequence[typing.SupportsFloat], arg12: collections.abc.Sequence[typing.SupportsInt], arg13: collections.abc.Sequence[str], arg14: collections.abc.Sequence[typing.SupportsFloat], arg15: typing.SupportsInt) -> list[float]:
    """
    Get momentum operator matrix elements
    """
def get_Phase_Operators(arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[typing.SupportsFloat], arg2: collections.abc.Sequence[typing.SupportsFloat], arg3: collections.abc.Sequence[typing.SupportsInt], arg4: collections.abc.Sequence[typing.SupportsFloat], arg5: collections.abc.Sequence[typing.SupportsInt], arg6: collections.abc.Sequence[str], arg7: collections.abc.Sequence[str], arg8: collections.abc.Sequence[typing.SupportsFloat], arg9: collections.abc.Sequence[typing.SupportsFloat], arg10: collections.abc.Sequence[typing.SupportsInt], arg11: collections.abc.Sequence[typing.SupportsFloat], arg12: collections.abc.Sequence[typing.SupportsInt], arg13: collections.abc.Sequence[str], arg14: collections.abc.Sequence[typing.SupportsFloat], arg15: collections.abc.Sequence[typing.SupportsFloat]) -> list[...]:
    """
    Get phase operator matrix elements
    """
def get_Position_Operators(arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[typing.SupportsFloat], arg2: collections.abc.Sequence[typing.SupportsFloat], arg3: collections.abc.Sequence[typing.SupportsInt], arg4: collections.abc.Sequence[typing.SupportsFloat], arg5: collections.abc.Sequence[typing.SupportsInt], arg6: collections.abc.Sequence[str], arg7: collections.abc.Sequence[str], arg8: collections.abc.Sequence[typing.SupportsFloat], arg9: collections.abc.Sequence[typing.SupportsFloat], arg10: collections.abc.Sequence[typing.SupportsInt], arg11: collections.abc.Sequence[typing.SupportsFloat], arg12: collections.abc.Sequence[typing.SupportsInt], arg13: collections.abc.Sequence[str], arg14: collections.abc.Sequence[typing.SupportsFloat], arg15: typing.SupportsInt) -> list[float]:
    """
    Get position operator matrix elements
    """
def get_T_Matrix(arg0: collections.abc.Sequence[str], arg1: collections.abc.Sequence[typing.SupportsFloat], arg2: collections.abc.Sequence[typing.SupportsFloat], arg3: collections.abc.Sequence[typing.SupportsInt], arg4: collections.abc.Sequence[typing.SupportsFloat], arg5: collections.abc.Sequence[typing.SupportsInt], arg6: collections.abc.Sequence[str], arg7: collections.abc.Sequence[str], arg8: collections.abc.Sequence[typing.SupportsFloat], arg9: collections.abc.Sequence[typing.SupportsFloat], arg10: collections.abc.Sequence[typing.SupportsInt], arg11: collections.abc.Sequence[typing.SupportsFloat], arg12: collections.abc.Sequence[typing.SupportsInt], arg13: collections.abc.Sequence[str], arg14: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
    """
    Get transformation matrix
    """
def get_WFN_On_Grid(arg0: collections.abc.Sequence[typing.SupportsFloat], arg1: collections.abc.Sequence[typing.SupportsFloat], arg2: collections.abc.Sequence[str], arg3: collections.abc.Sequence[typing.SupportsFloat], arg4: collections.abc.Sequence[typing.SupportsFloat], arg5: collections.abc.Sequence[typing.SupportsInt], arg6: collections.abc.Sequence[typing.SupportsFloat], arg7: collections.abc.Sequence[typing.SupportsInt], arg8: collections.abc.Sequence[str], arg9: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
    """
    Get wavefunction on grid
    """
