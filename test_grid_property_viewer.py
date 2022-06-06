from dataclasses import dataclass
from tkinter import Grid
from GridProperty import GridProperty, GridPropertyViewer
from GridPropertyName import GridPropertyName
from GridPropertyUnit import GridPropertyUnit
from GridParameters import GridParameters

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def grid_property() -> GridProperty:
    """generates grid property object for use in tests"""
    name = GridPropertyName.DISK_RADIUS
    unit = GridPropertyUnit.ECLIPSE_DURATION
    data = np.arange(11 * 11 * 7).reshape((11, 11, 7)).astype(float)
    mask = data < 100
    grid_parameters = GridParameters(0, 1, 11, 0, 2, 11, 5, 4)
    
    gp = GridProperty(name, unit, data, grid_parameters)
    gp.set_mask(mask)

    return gp

# def validate_image_shape(image: plt.AxesImage, grid_property: GridProperty) -> bool:
#     """This function validates the image shape"""
#     image_extent = image.get_extent()
#     grid_property.grid_parameters.



def test_init_defaults(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = False
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data == grid_property.get_data(False))
    assert viewer.index == grid_property.data.shape[2] // 2
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)

def test_init_masked(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == grid_property.data.shape[2] // 2
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)
    assert viewer.image.get_extent() == ""


def test_init_index(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 0
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)



def test_init_axis0(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 0, 
        grid_property = grid_property, 
        masked = True,
        index = 0
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$y$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)


def test_init_axis1(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 1, 
        grid_property = grid_property, 
        masked = True,
        index = 0
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$x$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)

    
def test_init_axis2(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 0
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)

def test_init_axis0(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 0
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)

