"""
This python module contains all the custom exceptions used for BeyonCE.
"""

import numpy as np

class LoadError(Exception):
    """Used when loading a class instance from a directory fails."""

    def __init__(self, type_string: str, directory: str) -> None:
        """
        This method initialises the custom LoadError.
        
        Parameters
        ----------
        type_string : str
            Name of the class that should be instantiated.
        directory : str
            Directory from which the class instance should be loaded.
        """
        message = f"failed to load {type_string} from {directory}"
        super().__init__(message)

class InvalidShapeError(Exception):
    """Used when multiple arrays do not have the same shape."""

    def __init__(self, 
            names_list: list[str],
            arrays_list: list[np.ndarray]
        ) -> None:
        """
        This method initialises the custom InvalidShapeError.

        Parameters
        ----------
        names_list : list[str]
            List of the names of all the input arrays.
        arrays_list : list[np.ndarray]
            List of the input arrays.
        """
        message = ''
        for name, array in zip(names_list, arrays_list):
            message = message + f'{name} {str(array.shape)}, '
        message = message[:-2] + ' should all have the same shape.'
        super().__init__(message)

class InvalidDimensionsError(Exception):
    """Used when an array doesn't have the right number of dimensions."""

    def __init__(self, 
            array_name: str, 
            array: np.ndarray, 
            num_dimensions: int
        ) -> None:
        """
        This method initialises the custom InvalidDeminsionsError.
        
        Parmaters
        ---------
        array_name : str
            Name of the input array.
        array : np.ndarray
            Input array.
        num_dimensions : int
            Number of dimensions the input array should have.
        """
        message = (f'The {array_name} ({len(array.shape)} dimensions) should '
            f'have {num_dimensions} dimensions')
        super().__init__(message)