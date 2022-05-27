from __future__ import annotations
import os
import numpy as np
import validate
from errors import LoadError
from GridParameters import GridParameters


class GridDiagnostics:
    """
    This class contains all the diagnostic information for the grid. This is
    concerns how well the linear interpolation of the rf dimension works. It
    can be saved and loaded.
    """

    def __init__(self, grid_parameters: GridParameters) -> None:
        """
        This is the constructor for the disk grid parameter class.
        """
        self._fy_dict: dict = {}
        self._disk_radius_dict: dict = {}

        self._set_valid_key_parameters(grid_parameters)


    def __str__(self) -> str:
        """
        This method returns a representation string of the grid diagnostics
        class.

        Returns
        -------
        str_string : str
            Representation string of the grid diagnostics class.
        """
        str_string = self.__repr__()
        return str_string


    def __repr__(self) -> str:
        """
        This method returns a representation string of the grid diagnostics
        class.

        Returns
        -------
        repr_string : str
            Representation string of the grid diagnostics class.
        """
        lines: list[str] = [""]
        lines.append("Grid Diagnostics")
        lines.append(28 * "-")
        lines.append(f"diagnostics saved: {len(self._fy_dict.keys())}")
        
        repr_string = "\n".join(lines)
        return repr_string


    def _generate_key(self, y: float, x: float) -> None:
        """
        This method determines whether or not the (y, x) pair is allowed
        if allowed values are set.

        Parameters
        ----------
        y : float
            The value of y that should be associated with the key.
        x : float
            The value of x that should be associated with the key.

        Returns
        -------
        key : str
            Key for the diagnostic dictionaries of the form (y, x).
        """
        y = validate.number(y, "y")
        x = validate.number(x, "x")

        message = ""
        if np.sum(np.isclose(y, self._valid_y)) == 0:
            message = message + "y value is not allowed. "

        if np.sum(np.isclose(x, self._valid_x)) == 0:
            message = message + "x value is not allowed. "

        if message != "":
            raise ValueError(message[:-1])

        key = f"({y}, {x})"

        return key


    def _set_valid_key_parameters(self, 
            grid_parameters: GridParameters
        ) -> None:
        """
        This method is used to set the allowed values for the creating of
        diagnostic keys.
        
        Parameters
        ----------
        grid_parameters : GridParameters
            Instance containing the dx and dy parameters for the generation
            of keys for the diagnostic dictionaries
        """
        self._valid_x = grid_parameters.dx.flatten()
        self._valid_y = grid_parameters.dy.flatten()


    def get_diagnostic(self, 
            y: float, 
            x: float
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        This method is used to extract the diagnostic values from each 
        dictionary according to the key provided
        
        Parameters
        ----------
        y : float
            The value of y that should be associated with the key.
        x : float
            The value of x that should be associated with the key.
        
        Returns
        -------
        fy : np.ndarray
            The fy values that correspond to the provided key (stored in 
            fy_dict property).
        disk_radius : np.ndarray
            The disk radius values that correspond to the provided key (stored 
            in disk_radius_dict property).
        """
        key = self._generate_key(y, x)
        fy = self._fy_dict[key]
        disk_radius = self._disk_radius_dict[key]

        return fy, disk_radius


    def save_diagnostic(self, 
            y: float,
            x: float, 
            fy: np.ndarray, 
            disk_radius: np.ndarray
        ) -> None:
        """
        This method is used to save a diagnostic value to each dictionary
        
        Parameters
        ----------
        y : float
            The value of y that should be associated with the key.
        x : float
            The value of x that should be associated with the key.
        fy : np.ndarray
            The fy values that correspond to the provided key (stored in 
            fy_dict property).
        disk_radius : np.ndarray
            The disk radius values that correspond to the provided key (stored 
            in disk_radius_dict property).
        """
        fy = validate.array(fy, "fy", dtype="float64", num_dimensions=1)
        disk_radius = validate.array(disk_radius, "disk_radius", 
            dtype="float64", num_dimensions=1)
        validate.same_shape_arrays([fy, disk_radius], ["fy", "disk_radius"])

        key = self._generate_key(y, x)
        self._fy_dict[key] = fy
        self._disk_radius_dict[key] = disk_radius


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

        np.save(f"{directory}/fy_dict", self._fy_dict)
        np.save(f"{directory}/disk_radius_dict", self._disk_radius_dict)

        np.save(f"{directory}/valid_y", self._valid_y)
        np.save(f"{directory}/valid_x", self._valid_x)


    @classmethod
    def load(cls, directory: str) -> GridDiagnostics:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        grid_diagnostics : GridDiagnostics
            This is the loaded object.
        """
        directory = validate.string(directory, "directory")
        grid_parameters = GridParameters(0, 1, 1, 0, 1, 1, 2, 1)
        
        try:    
            grid_diagnostics = cls(grid_parameters)
            
            grid_diagnostics._fy_dict = np.load(f"{directory}/fy_dict.npy",
                allow_pickle=True).item()
            grid_diagnostics._disk_radius_dict = np.load(
                f"{directory}/disk_radius_dict.npy", allow_pickle=True).item()
            
            grid_diagnostics._valid_x = np.load(f"{directory}/valid_x.npy")
            grid_diagnostics._valid_y = np.load(f"{directory}/valid_y.npy")
        
        except Exception:
            raise LoadError("grid diagnostics", directory) 

        return grid_diagnostics