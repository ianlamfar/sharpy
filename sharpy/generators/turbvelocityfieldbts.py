import numpy as np
import scipy.interpolate as interpolate

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout


def interp_rectgrid_vectorfield(points, grid, vector_field, out_value, regularGrid=False, num_cores=1):
    # check: https://en.wikipedia.org/wiki/Trilinear_interpolation
    npoints = points.shape[0]
    output = np.zeros((npoints, 3))
    if regularGrid:
        length = np.zeros((3))
        npoints_grid = np.zeros((3), dtype=int)
        delta = np.zeros((3))
        for idim in range(3):
            length[idim] = grid[idim][-1] - grid[idim][0]
            npoints_grid[idim] = len(grid[idim])
            delta[idim] = length[idim]/(npoints_grid[idim] - 1)

    for ipoint in range(npoints):
        # Check if the point is outside the box
        isout = False
        for idim in range(3):
            if (points[ipoint, idim] > grid[idim][-1]) or (points[ipoint, idim] < grid[idim][0]):
                isout = True
                output[ipoint, :] = out_value
                break

        # If the point is in the grid
        if not isout:
            # Compute the position
            igrid = np.zeros((3,), dtype=int)
            if regularGrid:
                for idim in range(3):
                    igrid[idim] = int(np.ceil((points[ipoint, idim] - grid[idim][0])/delta[idim]))
            else:
                for idim in range(3):
                    while points[ipoint, idim] >= grid[idim][igrid[idim]]:
                        igrid[idim] += 1

            xvec = np.array([grid[0][igrid[0] - 1],
                             grid[0][igrid[0]    ]])
            xvec = np.concatenate((xvec, xvec, xvec, xvec))
            yvec = np.array([grid[1][igrid[1] - 1],
                             grid[1][igrid[1] - 1],
                             grid[1][igrid[1]    ],
                             grid[1][igrid[1]    ]])
            yvec = np.concatenate((yvec, yvec))
            zvec = np.ones((8))
            zvec[0:4] *= grid[2][igrid[2] - 1]
            zvec[4:8] *= grid[2][igrid[2]    ]

            A = np.zeros((8,8))
            A[:, 0] = np.ones((8,))
            A[:, 1] = xvec
            A[:, 2] = yvec
            A[:, 3] = zvec
            A[:, 4] = xvec*yvec
            A[:, 5] = xvec*zvec
            A[:, 6] = yvec*zvec
            A[:, 7] = xvec*yvec*zvec

            Ainv = np.linalg.inv(A)
            x = points[ipoint, 0]
            y = points[ipoint, 1]
            z = points[ipoint, 2]
            for idim in range(3):
                b = np.array([vector_field[idim, igrid[0] - 1, igrid[1] - 1, igrid[2] - 1],
                              vector_field[idim, igrid[0]    , igrid[1] - 1, igrid[2] - 1],
                              vector_field[idim, igrid[0] - 1, igrid[1]    , igrid[2] - 1],
                              vector_field[idim, igrid[0]    , igrid[1]    , igrid[2] - 1],
                              vector_field[idim, igrid[0] - 1, igrid[1] - 1, igrid[2]    ],
                              vector_field[idim, igrid[0]    , igrid[1] - 1, igrid[2]    ],
                              vector_field[idim, igrid[0] - 1, igrid[1]    , igrid[2]    ],
                              vector_field[idim, igrid[0]    , igrid[1]    , igrid[2]    ]
                             ])
                f = np.dot(Ainv, b)
                output[ipoint, idim] = f[0] + f[1]*x + f[2]*y + f[3]*z + f[4]*x*y + f[5]*x*z + f[6]*y*z + f[7]*x*y*z

    return output


@generator_interface.generator
class TurbVelocityFieldBts(generator_interface.BaseGenerator):
    r"""
    Turbulent Velocity Field Generator from TurbSim bts files

    ``TurbVelocitityFieldBts`` is a class inherited from ``BaseGenerator``

    The ``TurbVelocitityFieldBts`` class generates a velocity field based on
    the input from a bts file generated by TurbSim.
    https://nwtc.nrel.gov/TurbSim

    To call this generator, the ``generator_id = TurbVelocityField`` shall be used.
    This is parsed as the value for the ``velocity_field_generator``
    key in the desired aerodynamic solver's settings.

    """
    generator_id = 'TurbVelocityFieldBts'
    generator_classification = 'TurbVelocityFieldBts'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Output solver-specific information in runtime'

    settings_types['turbulent_field'] = 'str'
    settings_default['turbulent_field'] = None
    settings_description['turbulent_field'] = 'BTS file path of the velocity file'

    settings_types['new_orientation'] = 'str'
    settings_default['new_orientation'] = 'xyz'
    settings_description['new_orientation'] = 'New order of the axes'

    settings_types['u_fed'] = 'list(float)'
    settings_default['u_fed'] = np.zeros((3,))
    settings_description['u_fed'] = 'Velocity at which the turbulence field is fed into the solid'

    settings_types['u_out'] = 'list(float)'
    settings_default['u_out'] = np.zeros((3,))
    settings_description['u_out'] = 'Velocity to set for points outside the interpolating box'

    settings_types['case_with_tower'] = 'bool'
    settings_default['case_with_tower'] = False
    settings_description['case_with_tower'] = 'Does the SHARPy case will include the tower in the simulation?'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.interpolator = []
        self.bbox = None

        self.x_grid = None
        self.y_grid = None
        self.z_grid = None

        self.vel = None

        self.dist_to_recirculate = None
        self.gird_size_vec = None
        self.grid_size_ufed_dir = None

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)
        self.settings = self.in_dict

        self.x_grid, self.y_grid, self.z_grid, self.vel = self.read_turbsim_bts(self.settings['turbulent_field'], self.settings['case_with_tower'])
        if not self.settings['new_orientation'] == 'xyz':
            # self.settings['new_orientation'] = 'zyx'
            self.x_grid, self.y_grid, self.z_grid, self.vel = self.change_orientation(self.x_grid, self.y_grid, self.z_grid, self.vel, self.settings['new_orientation'])

        self.bbox = self.get_field_bbox(self.x_grid, self.y_grid, self.z_grid)
        if self.settings['print_info']:
            cout.cout_wrap('The domain bbox is:', 1)
            cout.cout_wrap(' x = [' + str(self.bbox[0, 0]) + ', ' + str(self.bbox[0, 1]) + ']', 1)
            cout.cout_wrap(' y = [' + str(self.bbox[1, 0]) + ', ' + str(self.bbox[1, 1]) + ']', 1)
            cout.cout_wrap(' z = [' + str(self.bbox[2, 0]) + ', ' + str(self.bbox[2, 1]) + ']', 1)

        self.dist_to_recirculate = 0.
        self.grid_size_vec = np.array([np.max(self.x_grid) - np.min(self.x_grid),
                                                   np.max(self.y_grid) - np.min(self.y_grid),
                                                   np.max(self.z_grid) - np.min(self.z_grid)])
        self.grid_size_ufed_dir = np.dot(self.grid_size_vec,
                                         self.settings['u_fed']/np.lingalg.norm(self.settings['u_fed']))

        # self.init_interpolator(x_grid, y_grid, z_grid, vel)

    def init_interpolator(self, x_grid, y_grid, z_grid, vel):

        pass
        # for ivel in range(3):
        #     self.interpolator.append(interpolate.RegularGridInterpolator((x_grid, y_grid, z_grid),
        #                                                         vel[ivel,:,:,:],
        #                                                         bounds_error=False,
        #                                                         fill_value=self.settings['u_out'][ivel]))

    def generate(self, params, uext):
        zeta = params['zeta']
        for_pos = params['for_pos']
        t = params['t']

        offset = self.settings['u_fed']*t
        if offset > self.grid_size_ufed_dir:
            self.dist_to_recirculate += self.grid_size_ufed_dir
        # Through "offstet" zeta can be modified to simulate the turbulence being fed to the solid
        # Usual method for wind turbines
        self.interpolate_zeta(zeta,
                              for_pos,
                              uext,
                              offset = -1.*offset + self.dist_to_recirculate)

    def interpolate_zeta(self, zeta, for_pos, u_ext, interpolator=None, offset=np.zeros((3))):
        # if interpolator is None:
        #     interpolator = self.interpolator

        for isurf in range(len(zeta)):
            _, n_m, n_n = zeta[isurf].shape

            # Reorder the coordinates
            points_list = np.zeros((n_m*n_n, 3))
            ipoint = 0
            for i_m in range(n_m):
                for i_n in range(n_n):
                    points_list[ipoint, :] = zeta[isurf][:, i_m, i_n] + for_pos[0:3] + offset
                    ipoint += 1

            # Interpolate
            list_uext = interp_rectgrid_vectorfield(points_list, (self.x_grid, self.y_grid, self.z_grid), self.vel, self.settings['u_out'], regularGrid=True, num_cores=1)

            # Reorder the values
            ipoint = 0
            for i_m in range(n_m):
                for i_n in range(n_n):
                    u_ext[isurf][:, i_m, i_n] = list_uext[ipoint, :]
                    ipoint += 1

    @staticmethod
    def read_turbsim_bts(fname, case_with_tower=False):

        # This post may be useful to understand the function:
        # https://wind.nrel.gov/forum/wind/viewtopic.php?t=1384

        dtype = np.dtype([
            ("id", np.int16),
            ("nz", np.int32),
            ("ny", np.int32),
            ("tower_points", np.int32),
            ("ntime_steps", np.int32),
            ("dz", np.float32),
            ("dy", np.float32),
            ("dt", np.float32),
            ("u_mean", np.float32),
            ("HubHt", np.float32),
            ("Zbottom", np.float32),
            ("u_slope_scaling", np.float32),
            ("u_offset_scaling", np.float32),
            ("v_slope_scaling", np.float32),
            ("v_offset_scaling", np.float32),
            ("w_slope_scaling", np.float32),
            ("w_offset_scaling", np.float32),
            ("n_char_description", np.int32),
            ("description", np.dtype((bytes, 73))),
            # ("data", np.dtype((bytes, 12865632)))
            ("data", np.dtype((bytes, 2)))
        ])

        fileContent = np.fromfile(fname, dtype=dtype)
        n_char_description = fileContent[0][17]
        nbytes_data = 2*3*fileContent[0][4]*fileContent[0][1]*fileContent[0][2]
        dtype = np.dtype([
            ("id", np.int16),
            ("nz", np.int32),
            ("ny", np.int32),
            ("tower_points", np.int32),
            ("ntime_steps", np.int32),
            ("dz", np.float32),
            ("dy", np.float32),
            ("dt", np.float32),
            ("u_mean", np.float32),
            ("HubHt", np.float32),
            ("Zbottom", np.float32),
            ("u_slope_scaling", np.float32),
            ("u_offset_scaling", np.float32),
            ("v_slope_scaling", np.float32),
            ("v_offset_scaling", np.float32),
            ("w_slope_scaling", np.float32),
            ("w_offset_scaling", np.float32),
            ("n_char_description", np.int32),
            ("description", np.dtype((bytes, n_char_description))),
            ("data", np.dtype((bytes, nbytes_data)))
        ])

        fileContent = np.fromfile(fname, dtype=dtype)

        dictionary = {}
        for i in range(len(fileContent.dtype.names)):
            dictionary[fileContent.dtype.names[i]] = fileContent[0][i]

        scaling = np.array([dictionary['u_slope_scaling'], dictionary['v_slope_scaling'], dictionary['w_slope_scaling']])
        offset = np.array([dictionary['u_offset_scaling'], dictionary['v_offset_scaling'], dictionary['w_offset_scaling']])

        # Checks
        # print("Case description: ", dictionary['description'])
        if dictionary['description'][-1] == ".":
            cout.cout_wrap(("WARNING: I think there is something wrong with the case description. The length is not %d characters" %  n_char_description), 3)
            # print("Input", dictionary['n_char_description'], "as the number of characters of the case description")

        # vel_aux = np.fromstring(dictionary['data'], dtype='>i2')
        vel_aux = np.fromstring(dictionary['data'], dtype=np.int16)
        vel = np.zeros((3,dictionary['ntime_steps'],dictionary['ny'],dictionary['nz']))

        counter = -1
        for ix in range(dictionary['ntime_steps']):
            for iz in range(dictionary['nz']):
                for iy in range(dictionary['ny']):
                    for ivel in range(3):
                        counter += 1
                        vel[ivel,-ix,iy,iz] = (vel_aux[counter] - offset[ivel])/scaling[ivel]

        # Generate the grid
        height = dictionary['dz']*(dictionary['nz'] - 1)
        width = dictionary['dy']*(dictionary['ny'] - 1)

        x_grid = np.linspace(-dictionary['ntime_steps'] + 1, 0, dictionary['ntime_steps'])*dictionary['dt']*dictionary['u_mean']
        y_grid = np.linspace(-width/2, width/2, dictionary['ny'])
        if case_with_tower:
            z_grid = np.linspace(dictionary['Zbottom'], dictionary['Zbottom'] + height, dictionary['nz'])
        else:
            z_grid = np.linspace(-height/2, height/2, dictionary['nz'])

        return x_grid, y_grid, z_grid, vel

    @staticmethod
    def change_orientation(old_xgrid, old_ygrid, old_zgrid, old_vel, new_orientation_input):
        old_grid = []
        old_grid.append(old_xgrid.copy())
        old_grid.append(old_ygrid.copy())
        old_grid.append(old_zgrid.copy())
        new_orientation = ("%s." % new_orientation_input)[:-1]

        # Generate information for new_orientation
        if not old_vel.shape[0] == 3:
            print("ERROR: velocity must have three dimension")
        if (not (len(old_vel[0,:,0,0]) == len(old_xgrid))) or (not (len(old_vel[0,0,:,0]) == len(old_ygrid))) or (not (len(old_vel[0,0,0,:]) == len(old_zgrid))):
            print("ERROR: dimensions mismatch")
            return

        old_dim = np.array([len(old_xgrid),len(old_ygrid),len(old_zgrid)])
        position_in_old = np.zeros((3), dtype=int)
        sign = np.array([1,1,1], dtype=int)
        for ivel in range(3):
            if new_orientation[0] == "-":
                sign[ivel] = -1
                new_orientation = new_orientation[1:]
            if new_orientation[0] == "x":
                position_in_old[ivel] = 0
            elif new_orientation[0] == "y":
                position_in_old[ivel] = 1
            elif new_orientation[0] == "z":
                position_in_old[ivel] = 2

            new_orientation = new_orientation[1:]
        # print("position_in_old: ", position_in_old)

        # Check the new orientation system
        new_ux = np.zeros((3), dtype=int)
        new_uy = np.zeros((3), dtype=int)
        new_uz = np.zeros((3), dtype=int)

        new_ux[position_in_old[0]] = sign[0]
        new_uy[position_in_old[1]] = sign[1]
        new_uz[position_in_old[2]] = sign[2]

        aux_ux = np.cross(new_uy, new_uz)
        aux_uy = np.cross(new_uz, new_ux)
        aux_uz = np.cross(new_ux, new_uy)

        if (not (np.abs(aux_ux - new_ux) < 1e-9).all()) or (not (np.abs(aux_uy - new_uy) < 1e-9).all()) or (not (np.abs(aux_uz - new_uz) < 1e-9).all()):
            print("ERROR: The new coordinate system is not coherent")
            print("ux error: ", aux_ux - new_ux)
            print("uy error: ", aux_uy - new_uy)
            print("uz error: ", aux_uz - new_uz)

        # Output variables
        new_grid = [None]*3
        new_dim = old_dim[position_in_old]

        for ivel in range(3):
            new_grid[ivel] = old_grid[position_in_old[ivel]]*sign[ivel]
            if sign[ivel] == -1:
                new_grid[ivel] = new_grid[ivel][::-1]

        new_vel = np.zeros((3,new_dim[0],new_dim[1],new_dim[2]))
        # print("old_dim:", old_dim)
        # print("new_dim:", new_dim)
        # print("old_vel.shape:", old_vel.shape)
        # print("new_vel.shape:", new_vel.shape)

        # These loops will index variables associated with the new grid
        for ix in range(new_dim[0]):
            for iy in range(new_dim[1]):
                for iz in range(new_dim[2]):
                    new_i = np.array([ix, iy, iz])
                    # old_i = np.array([new_i[position_in_old[0]]*sign[0], new_i[position_in_old[1]]*sign[1], new_i[position_in_old[2]]*sign[2]])
                    old_i = np.array([new_i[position_in_old[0]], new_i[position_in_old[1]], new_i[position_in_old[2]]])
                    for icoord in range(3):
                        if sign[icoord] == -1:
                            old_i[icoord] = -1*old_i[icoord] - 1
                    for ivel in range(3):
                        # print("moving: ", position_in_old[ivel], ix, iy, iz, "to: ", ivel,new_i[0],new_i[1],new_i[2])
                        new_vel[ivel,new_i[0],new_i[1],new_i[2]] = old_vel[position_in_old[ivel], old_i[0], old_i[1], old_i[2]]*sign[ivel]

        return new_grid[0], new_grid[1], new_grid[2], new_vel

    def get_field_bbox(self, x_grid, y_grid, z_grid):
        bbox = np.zeros((3, 2))
        bbox[0, :] = [np.min(x_grid), np.max(x_grid)]
        bbox[1, :] = [np.min(y_grid), np.max(y_grid)]
        bbox[2, :] = [np.min(z_grid), np.max(z_grid)]
        return bbox
