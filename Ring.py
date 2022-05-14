import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Ellipse, PathPatch

import validate

class Ring:
    """
    The ring class is a simple object that has ring parameters:
        inner_radius
        outer_radius
        transmission
        inclination
        tilt
    With the additional ability to get_patch for plotting.
    """
    
    def __init__(self, 
            inner_radius: float, 
            outer_radius: float, 
            transmission: float, 
            inclination: float = 0, 
            tilt: float = 0
        ) -> None:
        '''
        This is the constructor for the class taking in all the necessary 
        parameters.
        
        Parameters
        ----------
        inner_radius : float
            This is the inner radius of the ring [R*].
        outer_radius : float
            This is the outer radius of the ring [R*].
        transmission : float
            This is the transmission of the ring [-], from 0 to 1.
        inclination : float
            This is the inclination of the ring (relation between horizontal
            and vertical width in projection) [default = 0 deg].
        tilt : float
            This is the tilt of the ring (angle between the x-axis and the
            semi-major axis of the projected ring ellipse) [default = 0 deg].
        '''
        self.inner_radius = validate.number(inner_radius, 'inner_radius', 
            lower_bound=0.)
        self.set_outer_radius(outer_radius)
        self.set_transmission(transmission)
        self.inclination = validate.number(inclination, 'inclination', 
            lower_bound=0, upper_bound=90)
        self.tilt = validate.number(tilt, 'tilt', lower_bound=-180,
            upper_bound=180)

    def __str__(self) -> str:
        '''
        This is the print representation of the ring object.

        Returns
        -------
        print_ring_class : str
            This contains the string representation of the ring class.
        '''
        # get parameter strings
        inner_radius_string = (f'{self.inner_radius:.4f}').rjust(8)
        outer_radius_string = (f'{self.outer_radius:.4f}').rjust(8)
        transmission_string = (f'{self.transmission:.4f}').rjust(8)
        inclination_string = (f'{self.inclination:.4f}').rjust(9)
        tilt_string = (f'{self.tilt:.4f}').rjust(16)

        # write lines
        lines = ['']
        lines.append('============================')
        lines.append('***** RING INFORMATION *****')
        lines.append('============================\n')
        lines.append(f'Inner Radius: {inner_radius_string} [R*]')
        lines.append(f'Outer Radius: {outer_radius_string} [R*]')
        lines.append(f'Transmission: {transmission_string} [-]')
        lines.append(f'Inclination: {inclination_string} [deg]')
        lines.append(f'Tilt: {tilt_string} [deg]')
        lines.append('\n============================')

        # get print string
        print_ring_class = "\n".join(lines)

        return print_ring_class

    def set_inner_radius(self, inner_radius: float) -> None:
        '''
        This method sets the inner radius of the ring.

        Parameters
        ----------
        inner_radius : float
            This is the inner radius of the ring [R*].
        '''
        inner_radius = validate.number(inner_radius, 'inner_radius', 
            lower_bound=0., upper_bound=self.outer_radius, exclusive=True)

        # correct for pyppluss simulation of light curves
        if inner_radius == 0.:
            inner_radius = 1e-16

        self.inner_radius = inner_radius

    def set_outer_radius(self, outer_radius: float) -> None:
        '''
        This method sets the outer radius.

        Parameters
        ----------
        outer_radius : float
            This is the outer radius of the ring [R*].
        '''
        self.outer_radius = validate.number(outer_radius, 'outer_radius',
            lower_bound=self.inner_radius, exclusive=True)

    def set_transmission(self, transmission: float) -> None:
        '''
        This method sets the transmission of the ring.
        
        Parameters
        ----------
        transmission : float
            This is the transmission of the ring [-], from 0 to 1.
        '''
        self.transmission = validate.number(transmission, 'transmission', 
            lower_bound=0., upper_bound=1.)

    def get_patch(self, 
            x_shift: float = 0, 
            y_shift: float = 0, 
            face_color: str = 'black'
        ) -> None:
        '''
        This function has been edited from a function written by Matthew 
        Kenworthy. The variable names, comments and documentation have been 
        changed, but the functionality has not.

        Parameters
        ----------
        x_shift : float
            The x-coordinate of the centre of the ring [default = 0].
        y_shift : float
            The y-coordinate of the centre of the ring [default = 0].
        inclination : float
            This is the tip of the ring [deg], from 0 deg (face-on) to 90 deg
            (edge-on) [default = 0].
        tilt : float
            This is the CCW angle [deg] between the orbital path and the semi-major 
            axis of the ring [default = 0].
        face_color : str
            The color of the ring system components [default = 'black'].
        
        Returns
        -------
        patch : matplotlib.patch
            Patch of the ring with input parameters.
        '''
        x_shift = validate.number(x_shift, 'x_shift')
        y_shift = validate.number(y_shift, 'y_shift')
        face_color = validate.string(face_color, 'face_color')

        # get ring system and ring parameters
        inc = np.deg2rad(self.inclination)
        phi = np.deg2rad(self.tilt)
        opacity = 1 - self.transmission

        # centre location
        ring_centre = np.array([x_shift, y_shift])

        # get an Ellipse patch that has an ellipse defined with eight CURVE4
        # Bezier curves actual parameters are irrelevant - get_path() returns
        # only a normalised Bezier curve ellipse which we then subsequently 
        # transform
        ellipse = Ellipse((1, 1), 1, 1, 0)

        # get the Path points for the ellipse (8 Bezier curves with 3 
        # additional control points)
        vertices = ellipse.get_path().vertices
        codes = ellipse.get_path().codes

        # define rotation matrix
        rotation_matrix = np.array([[np.cos(phi),  np.sin(phi)], 
                                    [np.sin(phi), -np.cos(phi)]])

        # squeeze the circle to the appropriate ellipse
        outer_annulus = self.outer_radius * vertices * ([ 1., np.cos(inc)])
        inner_annulus = self.inner_radius * vertices * ([-1., np.cos(inc)])

        # rotate and shift the ellipses
        outer_ellipse = ring_centre + np.dot(outer_annulus, rotation_matrix)
        inner_ellipse = ring_centre + np.dot(inner_annulus, rotation_matrix)
        
        # produce the arrays neccesary to produce a new Path and Patch object
        ring_vertices = np.vstack((outer_ellipse, inner_ellipse))
        ring_codes = np.hstack((codes, codes))

        # create the Path and Patch objects
        ring_path  = Path(ring_vertices, ring_codes)
        ring_patch = PathPatch(ring_path, facecolor=face_color, 
            edgecolor=(0., 0., 0., 1.), alpha=opacity, lw=2)
        
        return ring_patch