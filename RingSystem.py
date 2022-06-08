# generic modules
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# type hints
from enum import Enum
from typing import Union
from matplotlib.patches import Patch

# beyonce modules
import validate
from Ring import Ring


class RingSystem:
    """
    This class governs the simple manipulations of a ring system object. It
    contains the ring system parameters:
        planet_radius
        inner_radii
        outer_radii
        transmissions
        inclination
        tilt
    It also allows for ring manipulation:
        add_ring
        remove_ring
        replace_ring
        split_rings
        merge_rings
    With an additional plotting function and a logger for operations.
    """

    def __init__(self, 
            planet_radius: float, 
            inner_radii: np.ndarray, 
            outer_radii: np.ndarray, 
            transmissions: np.ndarray, 
            inclination: float, 
            tilt: float, 
            logging_level: Enum = logging.INFO
        ) -> None:
        """
        This is the constructor for the class taking in all the necessary 
        parameters.
        
        Parameters
        ----------
        planet_radius : float
            This is the size of the anchor body of the ring system [R*].
        inner_radii : np.ndarray (float 1-D)
            These are the inner radii of the rings that form the ring system 
            [R*].
        outer_radii : np.ndarray (float 1-D)
            These are the outer radii of the rings that form the ring system 
            [R*].
        transmissions : np.ndarray (float 1-D)
            These are the transmissions of the rings that form the ring system
            [-], from 0 to 1.
        inclination : float
            This is the tip of the ring system [deg], from 0 (face-on) to 90 
            (edge-on).
        tilt : float
            This is the CCW angle between the orbital path and the semi-major 
            axis of the disk [deg].
        logging_level : Enum
            This is one of the standard logging levels
        """
        self.planet_radius = validate.number(planet_radius, "planet_radius", 
            lower_bound=0.)
        self.inclination = validate.number(inclination, "inclination", 
            lower_bound=0., upper_bound=90.)
        self.tilt = validate.number(tilt, "tilt", lower_bound=-180., 
            upper_bound=180.)
        
        self._set_rings(inner_radii, outer_radii, transmissions)
        self._set_logger(logging_level)


    def __str__(self) -> str:
        """
        This method is used to print all the information pertaining to the 
        ring system.

        Returns
        -------
        str_string : str
            This contains the string representation of the ring system class.
        """
        return self.__repr__()

    def __repr__(self) -> str:
        """
        This method is used to print all the information pertaining to the 
        ring system.

        Returns
        -------
        repr_string : str
            This contains the string representation of the ring system class.
        """
        # print header
        lines = []
        lines.append("\n============================================"
            "==========================")
        lines.append("********************** RING SYSTEM INFORMATION "
            "***********************")
        lines.append("=============================================="
            "========================\n")
        
        # geometric parameters
        lines.append("Geometric Parameters")
        lines.append("--------------------\n")
        planet_radius_string = (f"{self.planet_radius:.2f}").rjust(7)
        inclination_string = (f"{self.inclination:.2f}").rjust(9)
        tilt_string = (f"{self.tilt:.2f}").rjust(16)
        lines.append(f"Planet Radius: {planet_radius_string} [R*]")
        lines.append(f"Inclination: {inclination_string} [deg]")
        lines.append(f"Tilt: {tilt_string} [deg]")
        lines.append("\n")

        # gather ring information
        ring_number = 1

        # print ring information
        lines.append("Ring Parameters")
        lines.append("---------------\n")
        lines.append("Ring #     Inner Radius     Outer Radius     "
            "Transmission")
        
        for ring_data in zip(*self.get_rings_data()):
            inner_radius, outer_radius, transmission = ring_data
            ring_number_string = str(ring_number).rjust(4)
            inner_radius_string = (f"{inner_radius:.2f}").rjust(11)
            outer_radius_string = (f"{outer_radius:.2f}").rjust(11)
            transmission_string = (f"{transmission:.2f}").rjust(13)
            
            string_parameters = (ring_number_string, inner_radius_string, 
                outer_radius_string, transmission_string)
            
            lines.append("     ".join(string_parameters))

            ring_number += 1

        lines.append("")
        lines.append("=============================================="
            "========================")

        repr_string = "\n".join(lines)

        return repr_string

    def _set_logger(self, logging_level: Enum) -> None:
        """
        This method sets the logger for this class instance.
        
        Parameters
        ----------
        logging_level : int
            Determines the logging level used for this class instance.
        """
        # validate
        logging_level = validate.number(logging_level, "logging_level", 
            check_integer=True, lower_bound=10, upper_bound=50)

        # define logger
        logger = logging.getLogger(str(np.random.normal(0, 1)))
        logger.setLevel(logging_level)

        # define formatter
        format = "%(asctime)s - %(levelname)-8s - %(funcName)s: %(message)s"
        formatter = logging.Formatter(format)
        
        # define console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)
        console_handler.setFormatter(formatter)

        # add console handler
        logger.addHandler(console_handler)
        logger.propagate = False
        
        # set
        self.logger = logger

    def _print_changes(self, action: str, before: bool = False) -> str:
        """
        This method prints all the information on the ring parameters.

        Parameters
        ----------
        action : str
            Name of the action performed.
        before : bool
            Adds a line of "=" at the top of the print statement. Used for the
            state of the ring system before the action is performed.

        Returns
        -------
        print_changes : str
            str representation of the class with an additional Before/After
            header.
        """
        # set up lines
        lines = [""]

        # print ring information
        if before:
            lines.append("================================================="
                "=====================")
            lines.append((f"Before: {action}").center(66))
        else:
            lines.append((f"After: {action}").center(66))
        
        lines.append(self.__str__()[1:])
        
        print_changes = "\n".join(lines)

        return print_changes
            
    def _set_rings(self, 
            inner_radii: np.ndarray, 
            outer_radii: np.ndarray, 
            transmissions: np.ndarray
        ) -> None:
        """
        This method is used to convert inner_radii, outer_radii and 
        transmissions to the correct number of Ring objects and attach them to
        the RingSystem.
        
        Parameters
        ----------
        inner_radii : np.ndarray (float 1-D)
            These are the inner radii of the rings that form the ring system 
            [R*].
        outer_radii : np.ndarray (float 1-D)
            These are the outer radii of the rings that form the ring system 
            [R*].
        transmissions : np.ndarray (float 1-D)
            These are the transmissions of the rings that form the ring system 
            [-], from 0 to 1.
        """
        # validate
        inner_radii = validate.array(inner_radii, "inner_radii", 
            lower_bound=0, dtype="float64", num_dimensions=1)
        outer_radii = validate.array(outer_radii, "outer_radii",
            lower_bound=0, dtype="float64", num_dimensions=1)
        transmissions = validate.array(transmissions, "transmissions",
            lower_bound=0, upper_bound=1, num_dimensions=1, dtype="float64")
        
        arrays_list = [inner_radii, outer_radii, transmissions]
        names_list = ["inner_radii", "outer_radii", "transmissions"]
        validate.same_shape_arrays(arrays_list, names_list)

        # clear all rings
        self.rings: np.ndarray[Ring] = np.array([], dtype=Ring)
        
        # add rings
        rings_data = (inner_radii, outer_radii, transmissions)

        for ring_data in zip(*rings_data):
            new_ring = Ring(*ring_data, self.inclination, self.tilt)
            self.rings = np.append(self.rings, new_ring)

    def get_inner_radii(self) -> np.ndarray:
        """
        This method is used to retrieve the inner radii.
        
        Returns
        -------
        inner_radii : np.ndarray (float 1-D)
            These are the inner radii of the rings that form the ring system 
            [R*].
        """
        inner_radii = np.zeros_like(self.rings)
        
        for i, ring in enumerate(self.rings):
            inner_radii[i] = ring.inner_radius
        
        return inner_radii.astype(float)

    def get_outer_radii(self) -> np.ndarray:
        """
        This method is used to retrieve the outer radii.

        Returns
        -------
        outer_radii : np.ndarray (float 1-D)
            These are the outer radii of the rings that form the ring system 
            [R*].
        """
        outer_radii = np.zeros_like(self.rings)

        for i, ring in enumerate(self.rings):
            outer_radii[i] = ring.outer_radius

        return outer_radii.astype(float)

    def get_transmissions(self) -> np.ndarray:
        """
        This method is used to retrieve the transmissions.

        Returns
        -------
        transmissions : np.ndarray (float 1-D)
            These are the transmissions of the rings that form the ring system
            [-], from 0 to 1.
        """
        transmissions = np.zeros_like(self.rings)

        for i, ring in enumerate(self.rings):
            transmissions[i] = ring.transmission

        return transmissions.astype(float)

    def get_rings_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method is used to retrieve the inner radii, outer radii and 
        transmissions.

        Parameters
        ----------
        transiting : bool
            Whether to return all rings or just the rings that transit the
            star [default = False].
        
        Returns
        -------
        inner_radii : np.ndarray (float 1-D)
            These are the inner radii of the rings that form the ring system 
            [R*].
        outer_radii : np.ndarray (float 1-D)
            These are the outer radii of the rings that form the ring system 
            [R*].
        transmissions : np.ndarray (float 1-D)
            These are the transmissions of the rings that form the ring system
            [-], from 0 to 1.
        """
        inner_radii = self.get_inner_radii()
        outer_radii = self.get_outer_radii()
        transmissions = self.get_transmissions()

        return inner_radii, outer_radii, transmissions

    def get_num_rings(self) -> int:
        """
        This method retrieves the number of rings.
        
        Returns
        -------
        num_rings : int
            The number of rings in the ring system.
        """
        return len(self.rings)

    def _prevent_ring_overlap(self, 
            index: int, 
            ring: Ring, 
            replace: bool = False
        ) -> None:
        """
        This method is used to ensure that the alterations made to the ring
        system don"t cause rings to overlap
        
        Parameters
        ----------
        index : int
            Index where the ring will be placed.
        ring : Ring
            Ring object that alters the ring system.
        replace : bool
            When replacing a ring, you should check the inner radius of the
            ring with index + 1 (not just index).
        """
        # validate input
        num_rings = self.get_num_rings()
        index = validate.number(index, "index", check_integer=True, 
            lower_bound=0, upper_bound=num_rings)
        ring = validate.class_object(ring, "ring", Ring, "Ring")

        # validate lower bound
        if index == 0:
            lower_bound = -np.inf
        else:
            lower_bound = self.rings[index - 1].outer_radius

        if ring.inner_radius < lower_bound:
            raise ValueError(f"Ring must have an inner radius "
                f"({ring.inner_radius:.2f}) > the preceding ring's "
                f"outer radius ({lower_bound:.2f})")

        # validate upper bound
        if replace:
            index += 1
        
        if index == num_rings:
            upper_bound = np.inf
        else:
            upper_bound = self.rings[index].inner_radius

        if ring.outer_radius > upper_bound:
            raise ValueError(f"Ring must have an outer radius "
                f"({ring.outer_radius:.2f}) less than the succeeding ring\'s"
                f" inner radius ({upper_bound:.2f})")

    def _override_ring_geometry(self, ring: Ring) -> Ring:
        """
        This method is used to override the ring geometry (inclination and 
        tilt) to the values of the ring system. The logger will log the change
        if it occurs.
        
        Parameters
        ----------
        ring : Ring
            This is a ring object used in a ring system mutation.

        Returns
        -------
        ring : Ring
            This is a ring object used in a ring system mutation that has the
            geometry overridden (if necessary)
        """
        if ring.inclination != self.inclination:
            self.logger.info(f"ring inclination overridden from "
                f"{ring.inclination:.2f} to {self.inclination:.2f}")

        if ring.tilt != self.tilt:
            self.logger.info(f"ring tilt overridden from {ring.tilt:.2f} to "
                f"{self.tilt:.2f}")

        ring.inclination = self.inclination
        ring.tilt = self.tilt
        
        return ring

    def add_ring(self, index: int, ring: Ring) -> None:
        """
        This method is used to add a ring at a given index. Note that the
        inclination and the tilt will be overridden by that of the ring
        system.

        Parameters
        ----------
        index : int
            This is the index of the ring where the ring will be placed. 
            Appending a ring is done with the index equal to the number of 
            rings.
        ring : Ring
            This is a ring object to be added the ring at the provided index.
        """
        # validate
        self._prevent_ring_overlap(index, ring)
        ring = self._override_ring_geometry(ring)
        debug_before = self._print_changes("add_ring()", before=True)

        self.rings = np.insert(self.rings, index, ring)

        debug_after = self._print_changes("add_ring()", before=False)
        self.logger.debug(debug_before + debug_after)
        self.logger.info(f"added ring at index {index}")

    def remove_ring(self, index: int) -> None:
        """
        This method is used to remove a ring to the rings property.
        
        Parameters
        ----------
        index : int
            This is the index of the ring that should be removed from the list
            of rings.
        """
        index = validate.number(index, "index", check_integer=True, 
            lower_bound=0, upper_bound=self.get_num_rings()-1)
        debug_before = self._print_changes("remove_ring()", before=True)

        self.rings = np.delete(self.rings, index)

        debug_after = self._print_changes("remove_ring()", before=False)
        self.logger.debug(debug_before + debug_after)
        self.logger.info(f"removed ring at index {index}")

    def replace_ring(self, index: int, ring: Ring) -> None:
        """
        This method is used to replace the ring at a given index. Note that
        the inclination and the tilt will be overridden by that of the ring
        system.

        Parameters
        ----------
        index : int
            This is the index of the ring that should be replaced from the 
            list of rings.
        ring : Ring
            This is a ring object to replace the ring at the provided index.
        """
        # validate
        index = validate.number(index, "index", check_integer=True, 
            lower_bound=0, upper_bound=self.get_num_rings()-1)
        self._prevent_ring_overlap(index, ring, replace=True)
        ring = self._override_ring_geometry(ring)
        debug_before = self._print_changes("replace_ring()", before=True)

        self.rings[index] = ring

        debug_after = self._print_changes("replace_ring()", before=False)
        self.logger.debug(debug_before + debug_after)
        self.logger.info(f"replaced ring at index {index}")

    def split_rings(self, num_divisions: Union[int, np.ndarray]) -> None:
        """
        This method is used to divide the rings into ringlets. Each ring 
        object is separated into num_divisions number of ringlets with the
        same transmission.

        Parameters
        ----------
        num_divisions : int or np.ndarray (int)
            Number of ringlets to divide rings into.
        """
        # validate
        try:
            num_divisions = validate.number(num_divisions, "num_divisions", 
                check_integer=True, lower_bound=2)
            num_divisions *= np.ones(self.get_num_rings()).astype(int)
        except TypeError:
            num_divisions = validate.array(num_divisions, "num_divisions",
                dtype="int64", lower_bound=1, num_dimensions=1)
            validate.same_shape_arrays([num_divisions, self.rings], 
                ["num_divisions", "self.rings"])

        debug_before = self._print_changes("split_rings()", before=True)

        divided_rings = np.zeros(int(np.sum(num_divisions))).astype(Ring)
        index = 0

        for ring, division in zip(self.rings, num_divisions):
            # for type hints
            ring: Ring
            
            delta_radius = (ring.outer_radius - ring.inner_radius) / division

            for i in range(division):
                ringlet_inner_radius = ring.inner_radius + i * delta_radius
                ringlet_outer_radius = ringlet_inner_radius + delta_radius
                
                ringlet = Ring(ringlet_inner_radius, ringlet_outer_radius, 
                    ring.transmission, self.inclination, self.tilt)
                
                divided_rings[index] = ringlet
                index += 1

        self.rings = divided_rings

        debug_after = self._print_changes("split_rings()", before=False)
        self.logger.debug(debug_before + debug_after)
        self.logger.info(f"split rings from {len(num_divisions)} to "
            f"{np.sum(num_divisions)}")

    def merge_rings(self, tolerance: float = 1e-6) -> None:
        """
        This method merges rings with the same transmission into one rings
        effectively reducing the number of rings to optimise and simulate.

        Parameters
        ----------
        tolerance : float
            This is the value by which the inner radius of the current ring
            can vary from the outer radius of previous ring. This value is
            necessary due to e.g. floating point representations.
        """
        # set up new initial conditions and rings array
        merged_rings = np.array([]).astype(Ring)

        inner_radius = self.rings[0].inner_radius
        outer_radius = self.rings[0].inner_radius
        transmission = self.rings[0].transmission

        debug_before = self._print_changes("merge_rings()", before=True)

        for ring_index, ring in enumerate(self.rings):
            adjacent = np.abs(ring.inner_radius - outer_radius) < tolerance
            
            if adjacent and ring.transmission == transmission:
                # grow the ring
                outer_radius = ring.outer_radius
            else:
                # create a new ring and append it
                merged_ring = Ring(inner_radius, outer_radius, transmission,
                    self.inclination, self.tilt)
                merged_rings = np.append(merged_rings, merged_ring)

                # set the new ring values
                inner_radius = ring.inner_radius
                outer_radius = ring.outer_radius
                transmission = ring.transmission

        merged_ring = Ring(inner_radius, outer_radius, transmission, 
            self.inclination, self.tilt)
        merged_rings = np.append(merged_rings, merged_ring)

        # scrap the first ring
        self.rings = merged_rings
        
        debug_after = self._print_changes("merge_rings()", before=False)
        self.logger.debug(debug_before + debug_after)
        self.logger.info(f"merged rings from {ring_index+1} to {len(self.rings)}")

    def _get_patches(self, 
            x_shift: float = 0, 
            y_shift: float = 0, 
            face_color: str = "black"
        ) -> list[Patch]:
        """
        This method is used to obtain a list of patches that characterise the
        ring system.
        
        Parameters
        ----------
        x_shift : float
            This is used to shift the centre of the ring system by some value
            in the x direction, as this might be useful for plotting purposes
            [default = 0].
        y_shift : float
            This is used to shift the centre of the ring system by some value
            in the y direction, as this might be useful for plotting purposes
            [default = 0].
        face_color : string
            The color of the ring system components [default = "black"].

        Returns
        -------
        patches : List (matplotlib.patches 1-D)
            This is a list of patches that characterise the ring system.
        """
        # validations
        x_shift = validate.number(x_shift, "x_shift")
        y_shift = validate.number(y_shift, "y_shift")
        face_color = validate.string(face_color, "face_color")

        # get ring system patches
        patches = []

        planet_patch = Circle((x_shift, y_shift), self.planet_radius, 
            facecolor=face_color)
        patches.append(planet_patch)

        for ring in self.rings:
            ring_patch = ring.get_patch(x_shift, y_shift, face_color)
            patches.append(ring_patch)

        return patches

    def plot(self, 
            ax: plt.Axes = None, 
            x_shift: float = 0, 
            y_shift: float = 0, 
            face_color: str = "black"
        ) -> plt.Axes:
        """
        This method is used to plot the RingSystem as is.

        Parameters
        ----------
        ax : plt.Axes
            Axes object that will contain the plot.
        x_shift : float
            This is used to shift the centre of the ring system by some value
            in the x direction, as this might be useful for plotting purposes
            [default = 0].
        y_shift : float
            This is used to shift the centre of the ring system by some value
            in the y direction, as this might be useful for plotting purposes
            [default = 0].
        face_color : str
            The color of the ring system components [default = "black"].

        Returns
        -------
        ax : matplotlib.Axes
            Axes object that contains the plot.
        """
        if ax is None:
            ax = plt.gca()
        ax = validate.class_object(ax, "ax", plt.Axes, "Axes")
        
        patches = self._get_patches(x_shift, y_shift, face_color)
        for patch in patches:
            ax.add_patch(patch)

        # plot settings
        ax.set_aspect("equal")
        ax.set_xlabel("$x [R_*]$")
        ax.set_ylabel("$y [R_*]$")

        # determine and set limits
        disk_radius = self.get_outer_radii()[-1]
        xlim = (x_shift - disk_radius - 2, x_shift + disk_radius + 2)
        ylim = (y_shift - disk_radius - 2, x_shift + disk_radius + 2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        return ax