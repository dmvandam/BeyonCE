from GridPropertyName import GridPropertyName
from GridPropertyUnit import GridPropertyUnit
from GridParameters import GridParameters
from GridProperty import GridProperty
from errors import LoadError, InvalidShapeError

import numpy as np
import pytest
import shutil


@pytest.fixture
def grid_property() -> GridProperty:
    grid_parameters = GridParameters(0, 1, 11, 0, 1, 11, 5, 4)
    data = np.arange(11*11*7).reshape((11, 11, 7)).astype(float)
    gp = GridProperty(
        name = GridPropertyName.DISK_RADIUS, 
        unit = GridPropertyUnit.ECLIPSE_DURATION, 
        data = data, 
        grid_parameters = grid_parameters
    )
    return gp


def test_str(grid_property: GridProperty) -> None:
    str_string = grid_property.__str__()
    assert str_string == ("\nDisk Radius [$t_{ecl}$]\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000\n\nGrid Parameters\n----------------------------"
        "\ndx:   0.00 ->   1.00 (11)\ndy:   0.00 ->   1.00 (11)\nrf:   "
        "5.00 ->   1.00 ->   5.00 (7)\ngrid_shape: (11, 11, 7)")


def test_repr(grid_property: GridProperty) -> None:
    repr_string = grid_property.__repr__()
    assert repr_string == ("\nDisk Radius [$t_{ecl}$]\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000")


def test_repr_mask(grid_property: GridProperty) -> None:
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    repr_string = grid_property.__repr__()

    assert repr_string == ("\nDisk Radius [$t_{ecl}$]\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000\n\nmask [out]:        100.0000%")


def test_set_mask_valid(grid_property: GridProperty) -> None:
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    assert np.all(grid_property.mask == mask)


def test_set_mask_invalid(grid_property: GridProperty) -> None:
    mask = np.ones_like((2, 2, 3)).astype(bool)
    
    with pytest.raises(InvalidShapeError) as ERROR:
        grid_property.set_mask(mask)
    
    message = ""
    names_list = ["mask", "data"]
    arrays_list = [mask, grid_property.data]
    
    for name, array in zip(names_list, arrays_list):
        message = message + f"{name} {str(array.shape)}, "
    message = message[:-2] + " should all have the same shape."
    
    assert str(ERROR.value) == message


def test_get_data_unmasked(grid_property: GridProperty) -> None:
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    data = grid_property.get_data(masked=False)
    
    assert np.all(data == grid_property.data)


def test_get_data_masked(grid_property: GridProperty) -> None:
    mask = grid_property.data < 37
    grid_property.set_mask(mask)
    data = grid_property.get_data(masked=True)
    
    expected_data = grid_property.data
    expected_data[mask] = np.nan
    
    assert np.all(np.isnan(data) == np.isnan(expected_data))


def test_get_data_masked_no_mask(grid_property: GridProperty) -> None:
    data = grid_property.get_data(masked=True)
    expected_data = grid_property.data
    assert np.all(data == expected_data)


def test_set_contrast_parameters_none(grid_property: GridProperty) -> None:
    assert grid_property.vmin == np.nanmin(grid_property.data)
    assert grid_property.vmax == np.nanmax(grid_property.data)
    assert grid_property.color_map == "viridis"
    assert grid_property.num_colors == 11


def test_set_contrast_parameters_all_nan(grid_property: GridProperty) -> None:
    grid_property.data = np.nan * grid_property.data
    grid_property.set_contrast_parameters()
    assert grid_property.vmin is None
    assert grid_property.vmax is None
    assert grid_property.color_map == "viridis"
    assert grid_property.num_colors == 11


def test_set_contrast_parameters(grid_property: GridProperty) -> None:
    grid_property.set_contrast_parameters(0, 2, "twilight", 3)
    assert grid_property.vmin == 0 
    assert grid_property.vmax == 2
    assert grid_property.color_map == "twilight"
    assert grid_property.num_colors == 3


def test_save_no_mask(grid_property: GridProperty) -> None:
    directory = "test_data"
    grid_property.save(directory)
    
    data = np.load(f"{directory}/{grid_property.name.name}_"
        f"{grid_property.unit.name}.npy")
    shutil.rmtree(directory)
    
    assert np.all(grid_property.data == data)


def test_save_with_mask(grid_property: GridProperty) -> None:
    directory = "test_data"
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    grid_property.save(directory)
    
    data = np.load(f"{directory}/{grid_property.name.name}_"
        f"{grid_property.unit.name}.npy")
    mask_load = np.load(f"{directory}/{grid_property.name.name}_"
        f"{grid_property.unit.name}_mask.npy")
    shutil.rmtree(directory)
    
    assert np.all(grid_property.data == data)
    assert np.all(grid_property.mask == mask_load)


def test_load_invalid() -> None:
    with pytest.raises(LoadError) as ERROR:
        GridProperty.load("invalid", 
            GridPropertyName.DISK_RADIUS, GridPropertyUnit.ECLIPSE_DURATION)

    message = "failed to load Disk Radius [Eclipse Duration] from invalid"
    assert str(ERROR.value) == message


def test_load_valid_no_mask(grid_property: GridProperty) -> None:
    directory = "test_data"
    grid_property.save(directory)
    
    grid_property_load = GridProperty.load(directory, 
        GridPropertyName.DISK_RADIUS, GridPropertyUnit.ECLIPSE_DURATION)

    assert np.all(grid_property.data == grid_property_load.data)
    assert grid_property_load.mask is None


def test_load_valid_with_mask(grid_property: GridProperty) -> None:
    directory = "test_data"
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    grid_property.save(directory)
    
    grid_property_load = GridProperty.load(directory, 
        GridPropertyName.DISK_RADIUS, GridPropertyUnit.ECLIPSE_DURATION)
    shutil.rmtree(directory)

    assert np.all(grid_property.data == grid_property_load.data)
    assert np.all(grid_property.mask == grid_property_load.mask)