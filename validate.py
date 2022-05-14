'''
This module is used to perform basic validations on arguments to methods
and functions. Method/function specific validations are defined inside the
methods/functions themselves.
'''

import numpy as np
from typing import Any, Union
from errors import InvalidDimensionsError, InvalidShapeError

def boolean(object: Any, object_name: str) -> bool:
    '''
    This function is used to raise an Exception if the object passed is not 
    a boolean.
    
    Parameters
    ----------
    object : Any
        Object that is validated as a boolean.
    object_name : str
        Name of the object to be passed in exception message.
    '''
    if type(object) not in [bool]:
        raise TypeError(f'The {object_name} argument must be a boolean')
    
    return object

def string(object: Any, object_name: str) -> str:
    '''
    This function is used to raise an Exception if the object passed is not
    a string.
    
    Parameters
    ----------
    object : Any
        Object that is validated as a string.
    object_name : str
        Name of the object to be passed in exception message.
    '''
    if type(object) != str:
        raise TypeError(f'The {object_name} argument must be a string')

    return object

def number(
        object: Any, 
        object_name: str, 
        check_integer: bool = False, 
        lower_bound: float = -np.inf, 
        upper_bound: float = np.inf, 
        exclusive: bool = False
    ) -> Union[float, int]:
    '''
    Parameters
    ----------
    object : Any
        Object that is validated as a number.
    object_name : str
        Name of the object to be passed in exception message.
    check_integer : bool
        Check if object is an integer [default = False].
    lower_bound : float
        Lower bound of the number [default = -np.inf].
    upper_bound : float
        Upper bound of the number [default = np.inf].
    exclusive : bool
        Determines whether to have exclusive bounds i.e. (lower_bound, 
        upper_bound), instead of [lower_bound, upper_bound]. 
    '''
    # validations
    object = _number_type(object, object_name, check_integer)
    object = _number_bounds(object, object_name, lower_bound, upper_bound, 
        exclusive)

    return object

def _number_type(
        object: Any, 
        object_name: str, 
        check_integer: bool = False
    ) -> Union[float, int]:
    '''
    This function is used to raise an Exception if the object passed is not a
    number.
    
    Parameters
    ----------
    object : Any
        Object that is validated as a number.
    object_name : str
        Name of the object to be passed in exception message.
    check_integer : bool
        Check if object is an integer [default = False].
    '''
    int_types = [int, np.int16, np.int32, np.int64]
    float_types = [float, np.float16, np.float32, np.float64, np.float128]

    if check_integer == True:
        if type(object) not in int_types:
            raise TypeError(f'The {object_name} argument must be an integer')

    else:
        if not (type(object) in int_types or type(object) in float_types):
            raise TypeError(f'The {object_name} argument must be a number')
    
    return object

def _number_bounds(
        object: Any, 
        object_name: str, 
        lower_bound: float = -np.inf, 
        upper_bound: float = np.inf,
        exclusive: bool = False
    ) -> Union[float, int]:
    '''
    This function validates whether the passed object agrees with the given 
    bounds.
    
    Parameters
    ----------
    object : Any
        Number to be bounded.
    object_name : str
        Name of the object to be passined in exception message.
    lower_bound : float
        Lower bound of the number [default = -np.inf].
    upper_bound : float
        Upper bound of the number [default = np.inf].
    exclusive : bool
        Determines whether to have exclusive bounds i.e. (lower_bound, 
        upper_bound), instead of [lower_bound, upper_bound]. 
    '''
    # validations
    _number_type(object, object_name)
    _number_type(lower_bound, 'lower_bound')
    _number_type(upper_bound, 'upper_bound')
    boolean(exclusive, 'exclusive')
    
    if lower_bound > upper_bound:
        raise ValueError(f'The upper_bound argument ({upper_bound:.4f}) must '
            f'be greater than the lower_bound ({lower_bound:.4f}) argument.')

    # validate number bounds
    raise_error = False
    message = f'The {object_name} argument must be '

    # inclusive
    lower = np.less
    upper = np.greater
    message_addition = 'than or equal to'
    
    # exclusive
    if exclusive:
        lower = np.less_equal
        upper = np.greater_equal
        message_addition = 'than'

    # try lower bound
    if lower(object, lower_bound):
        raise_error = True
        addition = f'greater {message_addition} {lower_bound:.4f} and '
        message = message + addition

    # try upper bound
    if upper(object, upper_bound):
        raise_error = True
        addition = f'less {message_addition} {upper_bound:.4f} and '
        message = message + addition

    if raise_error:
        raise ValueError(message[:-5])

    return object

def array(
        object: Any, 
        object_name: str, 
        lower_bound: float = None, 
        upper_bound: float = None, 
        exclusive: bool = False, 
        dtype: str = None,
        num_dimensions: int = None
    ) -> np.ndarray:
    '''
    This function is used to raise an Exception if the object passed is not 
    iterable array with the given restrictions.

    Parameters
    ----------
    object : Any
        Object to be checked for iterability.
    object_name : str
        Name of the object to be passed in Exception message.
    lower_bound : float
        Lower bound of the array [default = None].
    upper_bound : float
        Upper bound of the array [default = None].
    exclusive : bool
        Determines whether to have exclusive bounds i.e. (lower_bound, 
        upper_bound), instead of [lower_bound, upper_bound].
    dtype : str
        Type the array should have.
    num_dimensions : int
        Number of dimensions the array should have.
    
    '''
    # array?
    _array_like(object, object_name)

    # bounds?
    if (lower_bound is not None) or (upper_bound is not None):
        if lower_bound is None:
            lower_bound = -np.inf
        
        if upper_bound is None:
            upper_bound = np.inf

        _array_bounds(object, object_name, lower_bound, upper_bound, 
            exclusive)
    
    # dimensions?
    if num_dimensions is not None:
        _array_dimensions(object, object_name, num_dimensions)

    # array type?
    if dtype is not None:
        _array_type(object, object_name, dtype)

    return object

def _array_like(
        object: Any, 
        object_name: str
    ) -> np.ndarray:
    '''
    This function is used to raise an Exception if the object passed is not 
    iterable.
    
    Parameters
    ----------
    object : Any
        Object to be checked for iterability.
    object_name : str
        Name of the object to be passed in Exception message.
    '''
    if type(object) != np.ndarray:
        raise TypeError(f'The {object_name} argument must be array_like')
    
    return object

def _array_bounds(
        array: np.ndarray, 
        array_name: str, 
        lower_bound: float = -np.inf, 
        upper_bound: float = np.inf, 
        exclusive: bool = False
    ) -> np.ndarray:
    '''
    This function validates whether the passed array agrees with the given 
    bounds.

    Parameters
    ----------
    array: np.ndarray
        Array to be validated with lower and upper bounds.
    array_name : string
        Name of the array to be passed to exception message.
    lower_bound : float
        Lower bound of the array [default = -np.inf].
    upper_bound : float
        Upper bound of the array [default = np.inf].
    exclusive : bool
        Determines whether to have exclusive bounds i.e. (lower_bound, 
        upper_bound), instead of [lower_bound, upper_bound]. 
    '''
    # validate
    _array_like(array, array_name)
    _number_type(lower_bound, 'lower_bound')
    _number_type(upper_bound, 'upper_bound')
    boolean(exclusive, 'exclusive')

    if lower_bound >= upper_bound:
        raise ValueError(f'The upper_bound argument ({upper_bound:.4f}) must '
            f'be greater than the lower_bound ({lower_bound:.4f}) argument.')


    # validate array bounds
    raise_error = False
    message = f'The {array_name} argument must contain solely values '

    # inclusive
    lower = np.less
    upper = np.greater
    message_addition = 'than or equal to'
    
    # exclusive
    if exclusive:
        lower = np.less_equal
        upper = np.greater_equal
        message_addition = 'than'

    # try lower bound
    if np.any(lower(array, lower_bound)):
        raise_error = True
        addition = f'greater {message_addition} {lower_bound:.4f} and '
        message = message + addition

    # try upper bound
    if np.any(upper(array, upper_bound)):
        raise_error = True
        addition = f'less {message_addition} {upper_bound:.4f} and '
        message = message + addition

    if raise_error:
        raise ValueError(message[:-5])
    
    return array

def _array_dimensions(
        array: np.ndarray, 
        array_name: str, 
        num_dimensions: int
    ) -> np.ndarray:
    '''
    This function validates the number of dimension an array has.
    
    Parameters
    ----------
    array : np.ndarray
        Array to be validate for dimensions.
    array_name : str
        Name of the array to be passed to the exception message.
    num_dimensions : int
        Number of dimensions the array should have.
    '''
    # validations
    _array_like(array, array_name)
    _number_type(num_dimensions, 'num_dimensions', check_integer=True)

    # validate array dimensions
    array_dimensions = len(array.shape)

    if array_dimensions != num_dimensions:
        raise InvalidDimensionsError(array_name, array, num_dimensions)

    return array

def _array_type(array: np.ndarray, array_name: str, dtype: str) -> np.ndarray:
    '''
    This function validates the dtype of the input array.
    
    Parameters
    ----------
    array : np.ndarray
        Array to be validate for dimensions.
    array_name : str
        Name of the array to be passed to the exception message.
    dtype : str
        Type the array should have.
    '''
    # validate
    _array_like(array, array_name)
    string(dtype, 'dtype')

    # validate array type
    array_type = str(array.dtype)

    if array_type != dtype:
        raise TypeError(f'The {array_name} ({array_type}) argument should be'
            f' of type {dtype}.')

    return array

def same_shape_arrays(
        arrays_list: list[np.ndarray], 
        names_list: list[str]
    ) -> None:
    '''
    This method ensures that all the arrays in the list have the same length.
    
    Parameters
    ----------
    arrays_list : list (np.ndarray)
        List of arrays that will have their lengths compared.
    names_list : list (str)
        List of the names of each array.
    '''
    # validate
    if len(arrays_list) < 2:
        raise AttributeError('arrays_list argument should contain at least '
            'two arrays')

    if len(arrays_list) != len(names_list):
        raise AttributeError(f'arrays_list ({len(arrays_list)}) should have '
            f'the same length as names_list ({len(names_list)})')

    shape = arrays_list[0].shape
    error = False

    for array in arrays_list:
        if array.shape != shape:
            error = True
    
    if error:
        raise InvalidShapeError(names_list, arrays_list)

def class_object(
        object: Any, 
        object_name: str, 
        class_type: Any, 
        class_name: str
    ) -> Any:
    '''
    This method is used to ensure that the object passed is of the class type
    
    Parameters
    ----------
    object : Any
        Object that is validate against the class.
    object_name : str
        Name of the object to be passed in exception message.
    class_type : Any
        Class that the object is validated against.
    class_name : str
        Name of the class to be passed in exception message.

    Returns
    -------
    object : Any
        Object passed in.    
    '''
    if not isinstance(object, class_type):
        raise TypeError(f"{object_name} is not of type {class_name}")

    return object