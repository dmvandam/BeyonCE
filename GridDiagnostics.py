from __future__ import annotations
import numpy as np
import validate
from errors import LoadError


class GridDiagnostics:
    """
    This class contains all the diagnostic information for the grid. This is
    concerns how well the linear interpolation of the rf dimension works. It
    can be saved and loaded.
    """

    def __init__(self) -> None:
        """
        This is the constructor for the disk grid parameter class
        """
        self.fy_dict = {}
        self.disk_radius_dict = {}

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
        lines.append(f"diagnostics saved: {len(self.fy_dict.keys())}")
        
        repr_string = "\n".join(lines)
        return repr_string

    def get_diagnostic(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """
        This method is used to extract the diagnostic values from each 
        dictionary according to the key provided
        
        Parameters
        ----------
        key : str
            A key to identify a diagnostic value. The diagnostic value is
            composed of the y-index and the x-index (y, x).
        
        Returns
        -------
        fy : np.ndarray
            The fy values that correspond to the provided key (stored in 
            fy_dict property).
        disk_radius : np.ndarray
            The disk radius values that correspond to the provided key (stored 
            in disk_radius_dict property).
        """
        key = validate.string(key, "key")
        fy = self.fy_dict[key]
        disk_radius = self.disk_radius_dict[key]

        return fy, disk_radius

    def save_diagnostic(self, 
            key: str, 
            fy: np.ndarray, 
            disk_radius: np.ndarray
        ) -> None:
        """
        This method is used to save a diagnostic value to each dictionary
        
        Parameters
        ----------
        key : str
            A key to identify a diagnostic value. The diagnostic value is
            composed of the y-index and the x-index (y, x).
        fy : np.ndarray
            The fy values that correspond to the provided key (stored in 
            fy_dict property).
        disk_radius : np.ndarray
            The disk radius values that correspond to the provided key (stored 
            in disk_radius_dict property).
        """
        key = validate.string(key, "key")
        fy = validate.array(fy, "fy", dtype="float64", num_dimensions=1)
        disk_radius = validate.array(disk_radius, "disk_radius", 
            dtype="float64", num_dimensions=1)
        validate.same_shape_arrays([fy, disk_radius], ["fy", "disk_radius"])

        self.fy_dict[key] = fy
        self.disk_radius_dict[key] = disk_radius

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
        np.save(f"{directory}/fy_dict", self.fy_dict)
        np.save(f"{directory}/disk_radius", self.disk_radius_dict)

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

        try:
            grid_diagnostics = cls()
            grid_diagnostics.fy_dict = np.load(f"{directory}/fy_dict.npy",
                allow_pickle=True).item()
            grid_diagnostics.disk_radius_dict = np.load(
                f"{directory}/disk_radius_dict.npy", allow_pickle=True).item()
        except Exception:
            raise LoadError("grid diagnostics", directory)

        return grid_diagnostics