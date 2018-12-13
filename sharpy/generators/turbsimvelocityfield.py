import numpy as np
import scipy.interpolate as interpolate
import h5py as h5
import os
from xml.dom import minidom

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout


@generator_interface.generator
class TurbSimVelocityField(generator_interface.BaseGenerator):
    generator_id = 'TurbSimVelocityField'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['turbulent_field'] = 'str'
        self.settings_default['turbulent_field'] = None

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = 0.

        self.settings_types['offset'] = 'list(float)'
        self.settings_default['offset'] = np.zeros((3,))

        self.settings_types['centre_y'] = 'bool'
        self.settings_default['centre_y'] = True

        self.settings = dict()

        self.file = None
        self.extension = None
        self.turb_time = None
        self.turb_x_initial = None
        self.turb_y_initial = None
        self.turb_z_initial = None
        self.turb_u_ref = None
        self.turb_data = None

        self.bbox = None
        self.interpolator = 3*[None]

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        _, self.extension = os.path.splitext(self.settings['turbulent_field'])

        if self.extension is '.h5':
            self.read_btl(self.settings['turbulent_field'])
        if self.extension in '.xdmf':
            self.read_xdmf(self.settings['turbulent_field'])


    # these functions need to define the interpolators
    def read_btl(self, in_file):
        raise NotImplementedError('The BTL reader is not up to date!')
        # load the turbulent field HDF5
        with h5.File(self.settings['turbulent_field']) as self.file:
            # make time to increase from -t to 0 instead of 0 to t
            try:
                self.turb_time = self.file['time'].value
                self.turb_time = self.turb_time - np.max(self.turb_time)
                self.turb_u_ref = self.file['u_inf'].value
                self.turb_x_initial = self.turb_time*self.turb_u_ref + self.settings['offset'][0]
            except KeyError:
                self.turb_x_initial = self.file['x_grid'].value - np.max(self.file['x_grid'].value) + self.settings['offset'][0]
            self.turb_y_initial = self.file['y_grid'].value + self.settings['offset'][1]
            self.turb_z_initial = self.file['z_grid'].value + self.settings['offset'][2]

            self.turb_data = self.h5file['data/velocity'].value

            self.init_interpolator(self.turb_data, self.turb_x_initial, self.turb_y_initial, self.turb_z_initial)

    def read_xdmf(self, in_file):
        # store route of file for the other files
        route = os.path.dirname(os.path.abspath(in_file))

        # file to string
        with open(in_file, 'r') as self.file:
            data = self.file.read().replace('\n', '')

        # parse data
        from lxml import objectify
        tree = objectify.fromstring(data)

        # mesh dimensions
        dimensions = np.fromstring(tree.Domain.Topology.attrib['Dimensions'],
                                   sep=' ',
                                   count=3,
                                   dtype=int)

        # origin
        # NOTE: we can count here the offset?
        origin = np.fromstring(tree.Domain.Geometry.DataItem[0].text,
                               sep=' ',
                               count=int(tree.Domain.Geometry.DataItem[0].attrib['Dimensions']),
                               dtype=float)

        # dxdydz
        # because of how XDMF does it, it is actually dzdydx
        dxdydz = np.fromstring(tree.Domain.Geometry.DataItem[1].text,
                               sep=' ',
                               count=int(tree.Domain.Geometry.DataItem[1].attrib['Dimensions']),
                               dtype=float)
        # now onto the grid
        n_grid = len(tree.Domain.Grid.Grid)
        grid = [dict()]*n_grid
        for i, i_grid in enumerate(tree.Domain.Grid.Grid):
            # cycle through attributes
            for k_attrib, v_attrib in i_grid.attrib.items():
                grid[i][k_attrib] = v_attrib

        if n_grid > 1:
            cout.cout_wrap('CAREFUL: n_grid > 1, but we don\' support time series yet')

        # get Attributes (upper case A is not a mistake)
        for i_attrib, attrib in enumerate(i_grid.Attribute):
            grid[0][attrib.attrib['Name']] = dict()
            grid[0][attrib.attrib['Name']]['file'] = attrib.DataItem.text.replace(' ', '')

        # now we have the file names and the dimensions
        self.initial_x_grid = np.array(np.arange(0, dimensions[2]))*dxdydz[2]
        # z in the file is -y for us in sharpy (y_sharpy = right)
        self.initial_y_grid = -np.array(np.arange(0, dimensions[0]))*dxdydz[0]
        # y in the file is z for us in sharpy (up)
        self.initial_z_grid = np.array(np.arange(0, dimensions[1]))*dxdydz[1]

        # the domain now goes:
        # x \in [0, dimensions[0]*dx]
        # y \in [-dimensions[2]*dz, 0]
        # z \in [0, dimensions[1]*dy]

        centre_y_offset = 0.
        if self.settings['centre_y']:
            centre_y_offset = -0.5*(self.initial_y_grid[-1] - self.initial_y_grid[0])

        self.initial_x_grid += self.settings['offset'][0] + origin[0]
        self.initial_x_grid -= np.max(self.initial_x_grid)
        self.initial_y_grid += self.settings['offset'][1] + origin[1] + centre_y_offset
        self.initial_y_grid = self.initial_y_grid[::-1]
        self.initial_z_grid += self.settings['offset'][2] + origin[2]

        cout.cout_wrap('The domain bbox is:', 1)
        cout.cout_wrap(' x = [' + str(np.min(self.initial_x_grid)) + ', ' + str(np.max(self.initial_x_grid)) + ']', 1)
        cout.cout_wrap(' y = [' + str(np.min(self.initial_y_grid)) + ', ' + str(np.max(self.initial_y_grid)) + ']', 1)
        cout.cout_wrap(' z = [' + str(np.min(self.initial_z_grid)) + ', ' + str(np.max(self.initial_z_grid)) + ']', 1)

        # now we load the velocities (one by one, so we don't store all the
        # info more than once at the same time)
        velocities = ['ux', 'uz', 'uy']
        velocities_mult = np.array([1.0, -1.0, 1.0])
        for i_dim in range(3):
            file_name = grid[0][velocities[i_dim]]['file']

            # load file
            with open(route + '/' + file_name, "rb") as ufile:
                vel = np.fromfile(ufile, dtype=np.float64)

            vel = np.swapaxes(vel.reshape((dimensions[2], dimensions[1], dimensions[0]),
                                          order='F')*velocities_mult[i_dim], 1, 2)

            self.init_interpolator(vel,
                                   self.initial_x_grid,
                                   self.initial_y_grid,
                                   self.initial_z_grid,
                                   i_dim=i_dim)

    def generate(self, params, uext):
        zeta = params['zeta']
        for_pos = params['for_pos']
        # ts = params['ts']
        # dt = params['dt']
        t = params['t']

        self.interpolate_zeta(zeta,
                              for_pos,
                              uext)

    @staticmethod
    def get_bbox(zeta):
        """
        Calculates the bounding box of a given set of point coordinates
        :param zeta:
        :return:
        """
        bbox = np.zeros((3, 2))
        for i_surf in range(len(zeta)):
            ndim, nn, nm = zeta[i_surf].shape
            for idim in range(ndim):
                bbox[idim, :] = (min(bbox[idim, 0], np.min(zeta[i_surf][idim, :, :])),
                                 max(bbox[idim, 1], np.max(zeta[i_surf][idim, :, :])))
        return bbox

    def init_interpolator(self, data, x_grid, y_grid, z_grid, i_dim=None):
        if i_dim is None:
            for i_dim in range(3):
                self.interpolator[i_dim] = interpolate.RegularGridInterpolator((z_grid, y_grid, x_grid),
                                                                               data[i_dim, :, :, :],
                                                                               bounds_error=False,
                                                                               fill_value=0.0)
        else:
            self.interpolator[i_dim] = interpolate.RegularGridInterpolator((x_grid, y_grid, z_grid),
                                                                           data,
                                                                           bounds_error=False,
                                                                           fill_value=0.0)

    def interpolate_zeta(self, zeta, for_pos, u_ext):
        for i_dim in range(3):
            for isurf in range(len(zeta)):
                _, n_m, n_n = zeta[isurf].shape
                for i_m in range(n_m):
                    for i_n in range(n_n):
                        coord = zeta[isurf][:, i_m, i_n] + for_pos[0:3]
                        # coord = coord[::-1]
                        try:
                            u_ext[isurf][i_dim, i_m, i_n] = self.interpolator[i_dim](coord)
                        except ValueError:
                            print(coord)
                            raise ValueError()
                        # next = u_ext[isurf][i_dim, i_m, i_n]


