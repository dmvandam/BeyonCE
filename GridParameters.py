from __future__ import annotations

import os
import numpy as np

import validate
from errors import LoadError

class GridParameters:
    """
    This class contains all the information pertaining to the grid:
        dx (x-coordinate of disk centre w.r.t. eclipse centre [t_ecl])
        dy (y-coordinate of disk centre w.r.t. eclipse centre [t_ecl])
        rf (extent of rf_array in one-direction)
        rf_array (radius factor that compares with the smallest disk at a 
            given point)
        grid_shape (shape of the total grid)
        slice_shape (shape of dy, dx slice)
        extendable (whether the grid contains point (0, 0))
    It also contains methods to save and load the grid parameters.
    """

 
    def __init__(self, 
            min_x: float, 
            max_x: float, 
            num_x: int, 
            min_y: float, 
            max_y: float, 
            num_y: int, 
            max_rf: float, 
            num_rf: int
        ) -> None:
        """
        This is the constructor for the disk grid parameter class

        Parameters
        ----------
        num_xy : integer
            This is the resolution of the grid in the dx and dy directions.
        maximum_radius : float
            This is the maximum radius of the disk [t_ecl]
        num_rf : integer
            This is the resolution of the grid in the rf direction. Note that
            the size of this grid dimension is then (2 * num_rf - 1)
        maximum_rf : float
            This is the maximum rf value
        """
        self.dx = self._determine_dx(min_x, max_x, num_x)
        self.dy = self._determine_dy(min_y, max_y, num_y)
        self.rf, self.rf_array = self._determine_rf(max_rf, num_rf)
        
        self._set_grid_and_slice_shape()
        self._set_extendable()

 
    def __str__(self) -> str:
        """
        This returns the string representation of the class.

        Returns
        -------
        str_string : str
            String representation of the GridParameters class.
        """
        str_string = self.__repr__()
        return str_string

 
    def __repr__(self) -> str:
        """
        This generates a string representation of the grid parameters object. 
        
        Returns
        -------
        repr_string : str
            Representation string of the grid parameters class. 
        """
        dy, dx, rf_array = self.get_vectors()
        lines: list[str] = [""]
        lines.append("Grid Parameters")
        lines.append(28 * "-")
        
        dx_min = f"{dx[0]:.2f}".rjust(6)
        dx_max = f"{dx[-1]:.2f}".rjust(6)
        lines.append(f"dx: {dx_min} -> {dx_max} ({len(dx)})")

        dy_min = f"{dy[0]:.2f}".rjust(6)
        dy_max = f"{dy[-1]:.2f}".rjust(6)
        lines.append(f"dy: {dy_min} -> {dy_max} ({len(dy)})")

        rf_min = f"{1:.2f}".rjust(6)
        rf_max = f"{rf_array[0]:.2f}".rjust(6)
        rf_num = len(rf_array)
        lines.append(f"rf: {rf_max} -> {rf_min} -> {rf_max} ({rf_num})")

        lines.append(f"grid_shape: {str(self.grid_shape)}")

        repr_string = "\n".join(lines)
        return repr_string

 
    def _determine_dx(self, 
            min_x: float, 
            max_x: float, 
            num_x: int
        ) -> np.ndarray:
        """
        This method is used to determine the dx vector
        
        Parameters
        ----------
        min_x : float
            The minimum value of x [t_ecl].
        max_x : float
            The maximum value of x [t_ecl].
        num_x : int
            The number of dx elements.

        Returns
        -------
        dx : np.ndarray
            Grid dx dimension vector.
        """
        min_x = validate.number(min_x, "min_x")
        max_x = validate.number(max_x, "max_x")
        if min_x >= max_x:
            raise ValueError("max_x must be greater than min_x")
        num_x = validate.number(num_x, "num_x", check_integer=True, 
            lower_bound=1)
        
        dx = np.linspace(min_x, max_x, num_x)[None, :, None]
        return dx

 
    def _determine_dy(self, 
            min_y: float, 
            max_y: float, 
            num_y: int
        ) -> np.ndarray:
        """
        This method is used to determine the dx vector
        
        Parameters
        ----------
        min_y : float
            The minimum value of y [t_ecl].
        max_y : float
            The maximum value of y [t_ecl].
        num_y : int
            The number of dy elements.

        Returns
        -------
        dy : np.ndarray
            Grid dy dimension vector.
        """
        min_y = validate.number(min_y, "min_y")
        max_y = validate.number(max_y, "max_y")
        if min_y >= max_y:
            raise ValueError("max_y must be greater than min_y")
        num_y = validate.number(num_y, "num_y", check_integer=True, 
            lower_bound=1)
        
        dy = np.linspace(min_y, max_y, num_y)[:, None, None]
        return dy

 
    def _determine_rf(self, 
            max_rf: float, 
            num_rf: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        This method is used to determine the dx vector
        
        Parameters
        ----------
        max_rf : float
            The maximum value of rf [-].
        num_rf : int
            The number of rf elements (in one direction).

        Returns
        -------
        rf : np.ndarray
            Rf range from 1 to max_rf in num_rf.
        rf_array : np.ndarray
            Grid rf dimension vector.
        """
        max_rf = validate.number(max_rf, "max_rf", lower_bound=1)
        num_rf = validate.number(num_rf, "num_rf", check_integer=True,
            lower_bound=1)
        
        rf = np.linspace(1, max_rf, num_rf)
        rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
        return rf, rf_array

 
    def _set_grid_and_slice_shape(self) -> None:
        """
        This method sets useful grid parameters (grid shape and slice shape).
        """
        dy, dx, rf_array = self.get_vectors()
        self.grid_shape = (len(dy), len(dx), len(rf_array))
        self.slice_shape = (len(dy), len(dx))

 
    def _set_extendable(self) -> None:
        """
        This method is used to determine whether this particular set of grid
        parameters can be extended
        """
        dy, dx, _ = self.get_vectors()
        self.extendable: bool = dy[0] == 0 and dx[0] == 0

 
    def extend_grid(self) -> None:
        """
        This method is used to reflect the grid parameters about the x and y
        axes.
        """
        if not self.extendable:
            raise AttributeError("This grid parameter object can not be "
                "extended. That is only possible when dx[0] = dy[0] = 0.")
        num_y, num_x = self.slice_shape
        max_y = self.dy[-1, 0, 0]
        max_x = self.dx[0, -1, 0]
        
        self.dx = self._determine_dx(-max_x, max_x, 2 * num_x - 1)
        self.dy = self._determine_dy(-max_y, max_y, 2 * num_y - 1)
        self._set_grid_and_slice_shape()
        self._set_extendable()

 
    def get_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method returns the flattened dy, dx, and rf grid vectors.
        
        Returns
        -------
        dy : np.ndarray
            The y coordinates of the centre of the ellipse [t_ecl]
        dx : np.ndarray
            The x coordinates of the centre of the ellipse [t_ecl]
        rf_array : np.ndarray
            The rf radius stretch factors of the ellipse [-]
        """
        return self.dy.flatten(), self.dx.flatten(), self.rf_array

 
    def save(self, directory: str) -> None:
        """
        This method saves all the information of this object to a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
        """
        directory = validate.string(directory, "directory")
        
        if not os.path.exists(directory):
            os.mkdir(directory)

        np.save(f"{directory}/dx", self.dx)
        np.save(f"{directory}/dy", self.dy)
        np.save(f"{directory}/rf", self.rf)
        np.save(f"{directory}/rf_array", self.rf_array)

    @classmethod
 
    def load(cls, directory: str) -> GridParameters:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        grid_parameters : GridParameters
            This is the loaded object.
        """
        directory = validate.string(directory, "directory")

        try:
            grid_parameters = cls(0, 1, 1, 0, 1, 1, 2, 1)
            grid_parameters.dx = np.load(f"{directory}/dx.npy")
            grid_parameters.dy = np.load(f"{directory}/dy.npy")
            grid_parameters.rf = np.load(f"{directory}/rf.npy")
            grid_parameters.rf_array = np.load(f"{directory}/rf_array.npy")

            grid_parameters._set_grid_and_slice_shape()
            grid_parameters._set_extendable()
        except Exception:
            raise LoadError("grid parameters", directory)

        return grid_parameters