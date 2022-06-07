from GridProperty import GridProperty, GridPropertyViewer
from GridPropertyName import GridPropertyName
from GridPropertyUnit import GridPropertyUnit
from GridParameters import GridParameters

import matplotlib.pyplot as plt
import numpy as np
import pytest

class MockMouseEvent:
    def __init__(self, button: str) -> None:
        self.button = button


@pytest.fixture
def grid_property() -> GridProperty:
    """generates grid property object for use in tests"""
    name = GridPropertyName.DISK_RADIUS
    unit = GridPropertyUnit.ECLIPSE_DURATION
    data = np.arange(11 * 11 * 9).reshape((11, 11, 9)).astype(float)
    mask = data < 100
    grid_parameters = GridParameters(0, 1, 11, 0, 2, 11, 5, 5)
    
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
    assert viewer.image.get_extent() == (0.0, 1.0, 0.0, 2.0)


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
    assert viewer.axis == 0
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$y$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.dy.flatten())


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
    assert viewer.axis == 1
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$x$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.dx.flatten())

    
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


def test_init_axis2_scroll_up_successful(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 0
    )

    viewer.onscroll(MockMouseEvent("up"))

    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 1
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)


def test_init_axis2_scroll_up_unsuccessful(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 8
    )

    viewer.onscroll(MockMouseEvent("up"))

    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 8
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)


def test_init_axis2_scroll_down_successful(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 8
    )

    viewer.onscroll(MockMouseEvent("down"))

    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 7
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)


def test_init_axis2_scroll_down_unsuccessful(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 0
    )

    viewer.onscroll(MockMouseEvent("down"))

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


def test_init_frozen(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 0,
        frozen = True
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == True
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)


def test_init_frozen_scroll(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 0,
        frozen = True
    )
    
    viewer.onscroll(MockMouseEvent("up"))

    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 0
    assert viewer.coordinates is None
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == True
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)


def test_init_with_coodinates(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 8,
        coordinates = [(0.1, 0.3, 1.0), (0.2, 0.2, 3.0)]
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 8
    assert viewer.coordinates == [(0.1, 0.3, 1.0), (0.2, 0.2, 3.0)]
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)


def test_init_with_coodinates_tolerance_exceeded(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 4,
        coordinates = [(0.1, 0.3, 1.0), (0.2, 0.2, 3.0)]
    )
    assert viewer.ax == ax
    assert viewer.axis == 2
    assert np.all(viewer.data[~np.isnan(viewer.data)] == grid_property.get_data(True)[~np.isnan(grid_property.get_data(True))])
    assert viewer.index == 4
    assert viewer.coordinates == [(0.1, 0.3, 1.0), (0.2, 0.2, 3.0)]
    assert viewer.title_prefix == (f"{grid_property.name.get_name()} "
        f"[{grid_property.unit.get_unit()}]")
    assert viewer.frozen == False
    assert viewer.slice_name == "$R_f$"
    assert np.all(viewer.slice_values == 
        grid_property.grid_parameters.rf_array)


def test_init_with_invalid_coodinates(grid_property: GridProperty) -> None:
    ax = plt.gca()
    with pytest.raises(ValueError) as ERROR:
        GridPropertyViewer(
            ax = ax, 
            axis = 2, 
            grid_property = grid_property, 
            masked = True,
            index = 8,
            coordinates = [(12,)]
        )

    assert str(ERROR.value) == "all input coordinates must be tuples with three values"


def test_repr(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 0,
        frozen = True
    )

    repr_string = viewer.__repr__()

    lines: list[str] = [""]
    lines.append("Grid Property Viewer")
    lines.append(28 * "-")
    lines.append("Disk Radius [$t_{ecl}$] - $R_f$ = 5.0000 - horizontal")
    expected_string = "\n".join(lines)
    
    assert repr_string == expected_string


def test_str(grid_property: GridProperty) -> None:
    ax = plt.gca()
    viewer = GridPropertyViewer(
        ax = ax, 
        axis = 2, 
        grid_property = grid_property, 
        masked = True,
        index = 8,
        frozen = True
    )

    str_string = viewer.__str__()

    lines: list[str] = [""]
    lines.append("Grid Property Viewer")
    lines.append(28 * "-")
    lines.append("Disk Radius [$t_{ecl}$] - $R_f$ = 5.0000 - vertical")
    expected_string = "\n".join(lines)
    
    assert str_string == expected_string