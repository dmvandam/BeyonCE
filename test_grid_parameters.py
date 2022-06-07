from GridParameters import GridParameters
from errors import LoadError

import numpy as np
import pytest
import shutil
import os


def test_repr() -> None:
    gp = GridParameters(0, 1, 2, 0, 1, 2, 2, 2)
    repr_string = gp.__repr__()
    assert repr_string == ("\nGrid Parameters\n----------------------------\n"
        "dx:   0.00 ->   1.00 (2)\ndy:   0.00 ->   1.00 (2)\nrf:   2.00 ->   "
        "1.00 ->   2.00 (3)\ngrid_shape: (2, 2, 3)")


def test_str() -> None:
    gp = GridParameters(0, 1, 2, 0, 1, 2, 2, 2)
    str_string = gp.__str__()
    assert str_string == ("\nGrid Parameters\n----------------------------\n"
        "dx:   0.00 ->   1.00 (2)\ndy:   0.00 ->   1.00 (2)\nrf:   2.00 ->   "
        "1.00 ->   2.00 (3)\ngrid_shape: (2, 2, 3)")


def test_ymin_greater_than_ymax() -> None:
    with pytest.raises(ValueError) as ERROR:
        GridParameters(0, 1, 2, 2, 1, 2, 2, 2)
    assert str(ERROR.value) == "max_y must be greater than min_y"


def test_xmin_greater_than_xmax() -> None:
    with pytest.raises(ValueError) as ERROR:
        GridParameters(1, 0, 2, 0, 1, 2, 2, 2)
    assert str(ERROR.value) == "max_x must be greater than min_x"


def test_extendable_init() -> None:
    min_x = 0
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 4
    max_rf = 5
    num_rf = 4
    gp = GridParameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)
    dx = np.linspace(min_x, max_x, num_x)[None, :, None]
    dy = np.linspace(min_y, max_y, num_y)[:, None, None]
    rf = np.linspace(1, max_rf, num_rf)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    assert np.all(gp.dx == dx)
    assert np.all(gp.dy == dy)
    assert np.all(gp.rf == rf)
    assert np.all(gp.rf_array == rf_array)
    assert gp.grid_shape == (num_y, num_x, 2 * num_rf - 1)
    assert gp.slice_shape == (num_y, num_x)
    assert gp.extendable == True


def test_non_extendable_init() -> None:
    min_x = 0.5
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 4
    max_rf = 5
    num_rf = 4
    gp = GridParameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)
    dx = np.linspace(min_x, max_x, num_x)[None, :, None]
    dy = np.linspace(min_y, max_y, num_y)[:, None, None]
    rf = np.linspace(1, max_rf, num_rf)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    assert np.all(gp.dx == dx)
    assert np.all(gp.dy == dy)
    assert np.all(gp.rf == rf)
    assert np.all(gp.rf_array == rf_array)
    assert gp.grid_shape == (num_y, num_x, 2 * num_rf - 1)
    assert gp.slice_shape == (num_y, num_x)
    assert gp.extendable == False


def test_get_vectors() -> None:
    min_x = 0
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 5
    max_rf = 5
    num_rf = 4
    gp = GridParameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)
    dx = np.linspace(min_x, max_x, num_x)
    dy = np.linspace(min_y, max_y, num_y)
    rf = np.linspace(1, max_rf, num_rf)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    dy_get, dx_get, rf_array_get = gp.get_vectors()
    
    assert np.all(dx_get == dx)
    assert np.all(dy_get == dy)
    assert np.all(rf_array_get == rf_array)


def test_extend_grid_invalid() -> None:
    min_x = 0.5
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 4
    max_rf = 5
    num_rf = 4
    gp = GridParameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)

    with pytest.raises(AttributeError) as ERROR:
        gp.extend_grid()

    assert str(ERROR.value) == ("This grid parameter object can not be "
                "extended. That is only possible when dx[0] = dy[0] = 0.")

def test_extend_grid_valid() -> None:
    min_x = 0
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 5
    max_rf = 5
    num_rf = 4
    gp = GridParameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)
    gp.extend_grid()

    dx = np.linspace(-max_x, max_x, 2 * num_x - 1)[None, :, None]
    dy = np.linspace(-max_y, max_y, 2 * num_y - 1)[:, None, None]
    rf = np.linspace(1, max_rf, num_rf)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    assert np.all(gp.dx == dx)
    assert np.all(gp.dy == dy)
    assert np.all(gp.rf == rf)
    assert np.all(gp.rf_array == rf_array)
    assert gp.grid_shape == (2 * num_y - 1, 2 * num_x - 1, 2 * num_rf - 1)
    assert gp.slice_shape == (2 * num_y - 1, 2 * num_x - 1)
    assert gp.extendable == False

def test_save() -> None:
    min_x = 0
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 5
    max_rf = 5
    num_rf = 4
    gp = GridParameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)

    directory = "test_data"
    gp.save(directory)

    dx = np.load(f"{directory}/dx.npy")
    dy = np.load(f"{directory}/dy.npy")
    rf = np.load(f"{directory}/rf.npy")
    rf_array = np.load(f"{directory}/rf_array.npy")

    shutil.rmtree(directory)

    assert np.all(dx == gp.dx)
    assert np.all(dy == gp.dy)
    assert np.all(rf == gp.rf)
    assert np.all(rf_array == gp.rf_array)


def test_load_invalid() -> None:
    directory = "invalid directory"
    
    with pytest.raises(LoadError) as ERROR:
        GridParameters.load(directory)

    message = f"failed to load grid parameters from {directory}"
    assert str(ERROR.value) == message


def test_load_valid() -> None:
    directory = "test_data"
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    dx = np.linspace(0, 1, 2)
    dy = np.linspace(0, 2, 3)
    rf = np.linspace(1, 3, 2)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    np.save(f"{directory}/dx", dx)
    np.save(f"{directory}/dy", dy)
    np.save(f"{directory}/rf", rf)
    np.save(f"{directory}/rf_array", rf_array)

    gp = GridParameters.load(directory)
    shutil.rmtree(directory)

    assert np.all(dx == gp.dx)
    assert np.all(dy == gp.dy)
    assert np.all(rf == gp.rf)
    assert np.all(rf_array == gp.rf_array)
    assert gp.grid_shape == (3, 2, 3)
    assert gp.slice_shape == (3, 2)
    assert gp.extendable == True

