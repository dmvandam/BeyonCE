# type hints
from __future__ import annotations
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseEvent

# generic modules
import os
import numpy as np
import matplotlib.pyplot as plt

# beyonce modules
import validate
from errors import LoadError
from GridParameters import GridParameters
from GridPropertyName import GridPropertyName
from GridPropertyUnit import GridPropertyUnit

class GridProperty:
    """
    This class is used to define the following grid properties:
        disk radius
        inclination
        tilt
        fx map
        fy map
        diagnostic map
    Data manipulation only includes:
        set_mask (set some kind of mask based on the data)
    For plotting purposes:
        set_contrast_parameters (vmin, vmax, color_map and num_colors)
        plot_cube
        plot_slice
    And finally the data can be saved and loaded.
    """

    def __init__(self, 
            name: GridPropertyName, 
            unit: GridPropertyUnit, 
            data: np.ndarray, 
            grid_parameters: GridParameters
        ) -> None:
        """
        This is the class initialiser.
        
        Parameters
        ----------
        name : GridPropertyName
            The name of the disk property.
        unit : GridPropertyUnit
            The unit of the disk property.
        data : np.ndarray (float)
            The disk property data.
        grid_parameters : GridParameters
            The vectors that define each dimension of the data cube.
        """
        validate.class_object(name, 'name', GridPropertyName, 
            'GridpropertyName')
        validate.class_object(unit, 'unit', GridPropertyUnit, 
            'GridPropertyUnit')
        validate.class_object(grid_parameters, 'grid_parameters', 
            GridParameters, 'GridParameters')
        
        self.name = name
        self.unit = unit
        self.grid_parameters = grid_parameters
        self.data = validate.array(data, 'data', dtype='float64',
            num_dimensions=3)
        data_structure = np.ones(grid_parameters.grid_shape)
        validate.same_shape_arrays([data, data_structure], 
            ['data', 'data_structure'])
        
        self.set_contrast_parameters()

        self.mask = None

    def __str__(self) -> str:
        """
        This produces a string representation for the user.
        
        Returns
        -------
        str_string : str
            Representation string for disk property class.
        """
        data_shape = str(self.data.shape)
        str_string = self.__repr__() + f" with size {data_shape}"

        if self.mask is not None:
            str_string = str_string + " -> mask available"

        return str_string

    def __repr__(self):
        """
        This generates a string representation that combines the name
        and the unit.
        
        Returns
        -------
        repr_string : str
            This is the name and unit of the grid property.
        """
        name = self.name.get_name()
        unit = self.unit.get_unit()
        repr_string = f"{name} [{unit}]"

        return repr_string

    def set_mask(self, mask: np.ndarray) -> None:
        """
        This method is used to set a mask that can be applied to the data.
        Note that the True values of the mask are invalid and will be replaced
        with NaN's
        
        Parameters
        ----------
        mask : np.ndarray (bool)
            Mask that points to invalid values.
        """
        mask = validate.array(mask, 'mask', dtype='bool')
        validate.same_shape_arrays([mask, self.data], ['mask', 'data'])
        
        self.mask = mask

    def get_data(self, masked: bool = True) -> np.ndarray:
        """
        This method retrieves the data of the property.

        Parameters
        ----------
        masked : boolean
            Whether or not to convert invalid values to NaN's based on the
            available mask [default = True]
        
        Returns
        -------
        data : np.ndarray (float)
            The disk property data.
        """
        masked = validate.boolean(masked, 'masked')
        data = self.data

        if masked and self.mask is not None:
            data[self.mask] = np.nan
        
        return data

    def set_contrast_parameters(self, 
            vmin: float = None, 
            vmax: float = None,
            color_map: str = None, 
            num_colors: int = None
        ) -> None:
        """
        This method sets the contrast parameters for the plot_cube() and 
        plot_slice() methods.

        Parameters
        ----------
        vmin : float
            This is the lower limit of the color_map to adjust the contrast
            [default = None].
        vmax : float
            This is the upper limit of the color_map to adjust the contrast
            [default = None].
        color_map : string
            This is the name of the matplotlib color_map to be used to colour
            the image [default = 'viridis'].
        num_colors : integer
            This is the number of colors the color_map should be divided into.
            This is to make the image easier to interpret [default = 11].
        """
        all_nans = np.sum(np.isnan(self.data)) == np.prod(self.data.shape)
        
        if all_nans:
            self.vmin = None
            self.vmax = None
        else:
            if vmin is None:
                vmin = np.nanmin(self.data)
            self.vmin = validate.number(vmin, 'vmin')       
        
            if vmax is None:
                vmax = np.nanmax(self.data)
            self.vmax = validate.number(vmax, 'vmax')
        
        if color_map is None:
            color_map = 'viridis'
        self.color_map = validate.string(color_map, 'color_map')
        
        if num_colors is None:
            num_colors = 11
        self.num_colors = validate.number(num_colors, 'num_colors', 
            check_integer=True, lower_bound=2)

    def plot_cube(self, 
            axis: int = 2, 
            coordinates: list[tuple[float, float, float]] = None, 
            masked: bool = True
        ) -> None:
        """
        This method plots the value data cube.

        Parameters
        ----------
        axis : int
            This is the axis along which the cut will be made for the data
            cube. Note that these axes are ordered (dy, dx, rf) [default = 2].
        coordinates : list[tuple (float 3-D)]
            Contains the coordinates of the points to plot separately.
        masked : bool
            Determines whether or not the mask is applied to the data 
            [default = True].
        """
        fig, ax = plt.subplots()
        viewer = GridPropertyViewer(ax, axis, self, masked, 
            coordinates=coordinates)
        fig.canvas.mpl_connect('scroll_event', viewer.onscroll)
        plt.colorbar(viewer.image)
        plt.show()

    def plot_slice(self, 
            axis: int, 
            index: int, 
            ax: plt.Axes = None, 
            masked: bool = True
        ) -> tuple[plt.Axes, AxesImage]:
        '''
        This method allows you to plot a slice from the sjalot explorer cubes.
        It is designed to work with diskRadius, inclination, tilt and 
        gradients.

        Parameters
        ----------
        axis : integer
            This is the axis along which the cut will be made for the data
            cube. Note that these axes are ordered (dy, dx, fy).
        index : integer
            This is the index of the slice to plot.
        ax : matplotlib.Axes
            This is the object that will contain all the plotted information
            [default = None].
        masked : bool
            Determines whether or not the mask is applied to the data 
            [default = True].

        Returns
        -------
        ax : matplotlib.Axes
            This Axes objects contains all the plot information on it.
        image : mappable
            Used for colorbars.
        '''
        if ax is None:
            ax = plt.gca()
        ax = validate.class_object(ax, 'ax', plt.Axes, 'Axes')

        viewer = GridPropertyViewer(ax, axis, self, masked, 
            index=index, frozen=True)

        return viewer.ax, viewer.image

    def save(self, directory: str) -> None:
        """
        This method saves all the information of this object to a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
        """
        filename_data = f'{self.name.name}_{self.unit.name}'
        np.save(f'{directory}/{filename_data}', self.data)
        
        if self.mask is not None:
            filename_mask = f'{filename_data}_mask'
            np.save(f'{directory}/{filename_mask}', self.mask)

        self.grid_parameters.save(directory)

    @classmethod
    def load(cls, 
            directory: str, 
            name: GridPropertyName, 
            unit: GridPropertyUnit
        ) -> GridProperty:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        grid_property : GridProperty
            This is the loaded object.
        """
        directory = validate.string(directory, 'directory')
        validate.class_object(name, 'name', GridPropertyName, 
            'GridPropertyName')
        validate.class_object(unit, 'unit', GridPropertyUnit, 
            'GridPropertyUnit')
        
        try:
            filepath_data = f'{directory}/{name.name}_{unit.name}.npy'
            data = np.load(f'{filepath_data}')
            grid_parameters = GridParameters.load(directory)

            grid_property = cls(name, unit, data, grid_parameters)
            
            filepath_mask = f'{directory}/{name.name}_{unit.name}_mask.npy'
            if os.path.exists(f'{filepath_mask}'):
                mask = np.load(f'{filepath_mask}')
                grid_property.mask = mask
            
            grid_property.set_contrast_parameters()

        except Exception:
            type_string = f'{name.get_name()} [{unit.__str__()}]'
            raise LoadError(type_string, directory)

        return grid_property

class GridGradient(GridProperty):
    """
    This class is used to define the following grid gradients. Data 
    manipulation only includes:
        determine_mask (set some kind of mask based on a measured_gradient)
    For plotting purposes (from super class):
        set_contrast_parameters (vmin, vmax, color_map and num_colors)
        plot_cube
        plot_slice
    And finally the data can be saved and loaded.
    """

    def __init__(self, 
            data: np.ndarray, 
            grid_parameters: GridParameters,
            position: float
        ) -> None:
        """
        This is the class initialiser.
        
        Parameters
        ----------
        data : np.ndarray (float)
            The disk property data.
        grid_parameters : GridParameters
            The vectors that define each dimension of the data cube.
        position : float
            The position associated with this gradient.
        """
        super().__init__(
            GridPropertyName.GRADIENT, 
            GridPropertyUnit.STELLAR_LUMINOSITY_PER_DAY, 
            data, 
            grid_parameters
        )
        
        self.position = validate.number(position, 'position')
        self.measured_gradient = None

    def determine_mask(self, measured_gradient: float) -> None:
        """
        This method is used to determine the mask based on the appropriately
        scaled, measured gradient.

        Parameters
        ----------
        measured_gradient : float
            This value is the measured light curve gradient at this particular
            position.
        """
        self.measured_gradient = validate.number(measured_gradient, 
            'measured_gradient', lower_bound=0, upper_bound=1)

        self.mask = self.measured_gradient > self.data

    def save_gradient(self, directory: str) -> None:
        """
        This method saves all the information of this object to a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
        """
        # change the directory as there can be multiple gradients
        gradient_directory = f'{directory}/gradient_{self.position:.4f}'
        if not os.path.exists(gradient_directory):
            os.mkdir(gradient_directory)

        self.save(gradient_directory)

        # save additional gradient information
        np.save(f'{gradient_directory}/position', np.array([self.position]))

        if self.measured_gradient is not None:
            np.save(f'{gradient_directory}/measured_gradient', 
                np.array([self.measured_gradient]))
        
        self.grid_parameters.save(directory)

    @classmethod
    def load_gradient(cls, directory: str) -> GridGradient:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        grid_gradient : GridGradient
            This is the loaded object.
        """
        name = GridPropertyName.GRADIENT
        unit = GridPropertyUnit.STELLAR_LUMINOSITY_PER_DAY

        try:
            filepath_data = f'{directory}/{name.name}_{unit.name}.npy'
            data = np.load(f'{filepath_data}')
            
            grid_parameters = GridParameters.load(directory)
            position = np.load(f'{directory}/position.npy')[0]
            
            grid_gradient = cls(data, grid_parameters, position)
            
            filepath_measured_gradient = f'{directory}/measured_gradient.npy'
            if os.path.exists(filepath_measured_gradient):
                measured_gradient = np.load(filepath_measured_gradient)[0]
                grid_gradient.determine_mask(measured_gradient)

        except Exception:
            type_string = f'{name.get_name()} [{unit.__str__()}]'
            raise LoadError(type_string, directory)

        return grid_gradient


class GridPropertyViewer:
    """
    This class is used to view the grid properties and gradients either as a 
    cube or a single slice. Cubes can be viewed from any of the 3 axes and a
    list of coordinates can be provided to plot throughout the cube.
    """

    def __init__(self, 
            ax: plt.Axes, 
            axis: int, 
            grid_property: GridProperty,
            masked: bool, 
            index: int = None, 
            frozen: bool = False, 
            coordinates: list[tuple[float, float, float]] = None
        ) -> None:
        """
        Initialiser for the cube viewer.

        Parameters
        ----------
        ax : plt.Axes
            Axis object where plotting will occur.
        axis : int
            Which axis to use as the scrolling axis.
        grid_property : GridProperty
            Is used to obtain data, extent and contrast properties.
        masked : bool
            Use the masked data or the unmasked data.
        index : int
            Initial index to plot [default = None -> middle of cube].
        frozen : bool
            Determines whether the plot is scrollable [default = False].
        coordinates : list[tuple]
            List of 3-D coordinates to plot if visible in the given slice.
        """
        self.axis = validate.number(axis, 'axis', check_integer=True, 
            lower_bound=0, upper_bound=2)
        ax = validate.class_object(ax, 'ax', plt.Axes, 'Axes')
        grid_property = validate.class_object(grid_property, 'grid_property', 
            GridProperty, 'GridProperty')
        self.ax = ax
        self.data = grid_property.get_data(masked)
        self.slice_values = grid_property.grid_parameters.get_vectors()[axis]

        if index is None:
            index = len(self.slice_values) // 2
        self.index = validate.number(index, 'index', check_integer=True, 
            lower_bound=0, upper_bound=len(self.slice_values) - 1)

        if coordinates is not None:
            coordinates = validate.class_object(coordinates, 'coordinates',
                list, 'List')
            for coordinate in coordinates:
                if len(coordinate) != 3:
                    raise ValueError('all input coordinates must be tuples '
                        'with three values')
        self.coordinates = coordinates
        self.tolerance = np.abs(self.slice_values[1] - self.slice_values[0]) / 3

        self.slice_name = ['$y$', '$x$', '$R_f$'][axis]
        
        
        self.property_name = grid_property.__repr__()
            
        self.update_title()
        self.set_labels()
        cmap = plt.cm.get_cmap(grid_property.color_map, 
            grid_property.num_colors)
        extent = self.determine_extent(grid_property.grid_parameters)

        # get data slice and set image
        data_slice = np.take(self.data, self.index, axis)
        self.image = self.ax.imshow(data_slice, origin='lower', cmap=cmap, 
            vmin=grid_property.vmin, vmax=grid_property.vmax, extent=extent)

        self.set_rf_ticklabels(grid_property.grid_parameters)

        self.frozen = False
        self.update()
        if frozen is not None:
            self.frozen = validate.boolean(frozen, 'frozen')

    def update_title(self) -> None:
        """
        This method generates the title of the given slice
        """
        title = (f'{self.property_name} - {self.slice_name} = '
            f'{self.slice_values[self.index]:.4f}')
        
        if self.axis == 2: 
            minimum_radius_index = len(self.slice_values) // 2
            if self.index < minimum_radius_index:
                title = title + ' - horizontal'
            elif self.index > minimum_radius_index:
                title = title + ' - vertical'
        
        self.ax.set_title(title)

    def set_labels(self) -> None:
        """
        This method is used to generate the axes labels
        """
        xlabel = ['$R_f$', '$R_f$', '$x$'][self.axis]
        ylabel = ['$x$', '$y$', '$y$'][self.axis]
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def determine_extent(self, 
            grid_parameters: GridParameters
        ) -> tuple[float, float, float, float]:
        """
        This method determines the extent for the plot (rf is manipulated 
        because of its non-linear nature).
        
        Parameters
        ----------
        grid_parameters : GridParameters
            Contains all the information for the grid that is used to extract
            data.

        Returns
        -------
        extent : tuple (4-D)
            The extent for the matplotlib imshow image.
        """
        dx = grid_parameters.dx.flatten()
        x_extent = (dx[0], dx[-1])

        dy = grid_parameters.dy.flatten()
        y_extent = (dy[0], dy[-1])

        rf = grid_parameters.rf.flatten()
        rf_extent = (0, 2 * (rf[-1] - rf[0]))

        horizontal_extent = [rf_extent, rf_extent, x_extent][self.axis]
        vertical_extent = [x_extent, y_extent, y_extent][self.axis]

        extent = (*horizontal_extent, *vertical_extent)

        return extent

    def set_rf_ticklabels(self, grid_parameters: GridParameters) -> None:
        """
        This method sets the labels for the x and y axes. Note that this can
        not be done with extent due to the fact that Rf is not a linear axis
        (rf_max -> 1 -> rf_max)
        
        Parameters
        ----------
        grid_parameters : GridParameters
            Contains all the information for the grid that is used to extract
            data.
        """
        if self.axis == 2:
            return
        
        rf = grid_parameters.rf
        rf_range = rf[-1] - rf[0]
        
        locations = self.ax.get_xticks()
        
        labels = 1 + np.abs(locations - rf_range)
        labels = np.char.mod('%.2f', labels)

        self.ax.set_xticks(locations)
        self.ax.set_xticklabels(labels)

    def onscroll(self, event: MouseEvent) -> None:
        '''
        This method determines what happens what happens when the scroll wheel
        is used to navigate through the data cube.
        
        Parameters
        ----------
        event : MouseEvent
            This is used to register scroll events.
        '''
        if event.button == 'up' and self.index < len(self.slice_values) - 1:
            self.index += 1
        elif self.index > 0:
            self.index -= 1

        self.update()

    def update(self) -> None:
        '''
        This method is used to update the slice of the data cube being viewed.
        '''
        if self.frozen:
            return

        data_slice = np.take(self.data, self.index, self.axis)
        self.image.set_data(data_slice)
        self.plot_coordinates()
        
        self.update_title()

        self.image.axes.figure.canvas.draw()

    def plot_coordinates(self) -> None:
        """
        This method is used to plot a point at a specified location.
        """
        if self.coordinates is None:
            return

        for coordinate in self.coordinates:
            distance = np.abs(self.slice_values[self.index] - 
                coordinate[self.axis])

            if self.tolerance > distance:
                y, x = np.delete(coordinate, self.axis)
                self.ax.plot(x, y, 'ro')
            else:
                self.ax.lines = []