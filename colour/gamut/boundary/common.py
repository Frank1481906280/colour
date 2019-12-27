# -*- coding: utf-8 -*-
"""
Common Gamut Boundary Descriptor (GDB) Utilities
================================================

Defines various *Gamut Boundary Descriptor (GDB)* common utilities.

-   :func:`colour.unsparse_gamut_boundary_descriptor`

See Also
--------
`Gamut Boundary Descriptor Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/gamut/boundary.ipynb>`_

References
----------
-   :cite:`` :
"""

from __future__ import division, unicode_literals

import numpy as np
import scipy.interpolate

from colour.constants import DEFAULT_INT_DTYPE
from colour.models import Jab_to_JCh, JCh_to_Jab

from colour.utilities import (as_float_array, as_int_array,
                              is_trimesh_installed, orient, tsplit, tstack,
                              warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'close_gamut_boundary_descriptor', 'unsparse_gamut_boundary_descriptor',
    'tessellate_gamut_boundary_descriptor'
]


def close_gamut_boundary_descriptor(GDB_m, Jab, E=np.array([50, 0, 0])):

    GDB_m = np.copy(as_float_array(GDB_m))
    Jab = as_float_array(Jab)
    E = as_float_array(E)

    if not np.allclose(GDB_m[0, ...], GDB_m[0, ...][0]):
        JCh_l = np.mean(
            Jab_to_JCh(Jab[Jab[..., 0] == np.min(Jab[..., 0])] - E), axis=0)

        warning(
            'Inserting a singularity at the bottom of GBD: {0}'.format(JCh_l))
        GDB_m = np.insert(
            GDB_m, 0, np.tile(JCh_l, [1, GDB_m.shape[1], 1]), axis=0)

    if not np.allclose(GDB_m[-1, ...], GDB_m[-1, ...][0]):
        JCh_h = np.mean(
            Jab_to_JCh(Jab[Jab[..., 0] == np.max(Jab[..., 0])] - E), axis=0)

        warning('Inserting a singularity at the top of GBD: {0}'.format(JCh_h))

        GDB_m = np.insert(
            GDB_m,
            GDB_m.shape[0],
            np.tile(JCh_h, [1, GDB_m.shape[1], 1]),
            axis=0)

    return GDB_m


def unsparse_gamut_boundary_descriptor(GDB_m):
    GDB_m = as_float_array(GDB_m)

    GDB_m_i = np.copy(GDB_m)
    shape_r, shape_c = GDB_m.shape[0], GDB_m.shape[1]

    r_slice = np.s_[0:shape_r]
    c_slice = np.s_[0:shape_c]

    # If bounding columns have NaN, :math:`GDB_m` matrix is tiled
    # horizontally so that right values interpolate with left values and
    # vice-versa.
    if np.any(np.isnan(GDB_m[..., 0])) or np.any(np.isnan(GDB_m[..., -1])):
        warning(
            'Gamut boundary descriptor matrix bounding columns contains NaN '
            'and will be horizontally tiled!')
        c_slice = np.s_[shape_c:shape_c * 2]
        GDB_m_i = np.hstack([GDB_m] * 3)

    # If bounding rows have NaN, :math:`GDB_m` matrix is reflected vertically
    # so that top and bottom values are replicated via interpolation, i.e.
    # equivalent to nearest-neighbour interpolation.
    if np.any(np.isnan(GDB_m[0, ...])) or np.any(np.isnan(GDB_m[-1, ...])):
        warning('Gamut boundary descriptor matrix bounding rows contains NaN '
                'and will be vertically reflected!')
        r_slice = np.s_[shape_r:shape_r * 2]
        GDB_m_f = orient(GDB_m_i, 'Flop')
        GDB_m_i = np.vstack([GDB_m_f, GDB_m_i, GDB_m_f])

    mask = np.any(~np.isnan(GDB_m_i), axis=-1)
    x = np.linspace(0, 1, GDB_m_i.shape[0])
    y = np.linspace(0, 1, GDB_m_i.shape[1])
    x_g, y_g = np.meshgrid(x, y, indexing='ij')
    values = GDB_m_i[mask]
    for i in range(3):
        GDB_m_i[..., i] = scipy.interpolate.griddata(
            (x_g[mask], y_g[mask]),
            values[..., i], (x_g, y_g),
            method='linear')

    print(GDB_m_i.shape)
    print(r_slice, c_slice)
    return GDB_m_i[r_slice, c_slice, :]


def interpolate_gamut_boundary_descriptor(GDB_m, m, n):
    GDB_m = as_float_array(GDB_m)

    GDB_m_i = np.zeros([m, n, 3])

    x = np.linspace(0, 1, GDB_m.shape[0])
    y = np.linspace(0, 1, GDB_m.shape[1])
    x_g, y_g = np.meshgrid(x, y, indexing='ij')

    x_s = np.linspace(0, 1, m)
    y_s = np.linspace(0, 1, n)
    x_s_g, y_s_g = np.meshgrid(x_s, y_s, indexing='ij')
    for i in range(3):
        GDB_m_i[..., i] = scipy.interpolate.griddata(
            (np.ravel(x_g), np.ravel(y_g)),
            np.ravel(GDB_m[..., i]), (x_s_g, y_s_g),
            method='linear')

    return GDB_m_i


def tessellate_gamut_boundary_descriptor(GDB_m):
    if is_trimesh_installed():
        import trimesh

        vertices = JCh_to_Jab(GDB_m)

        # Wrapping :math:`GDB_m` to create faces between the outer columns.
        vertices = np.insert(
            vertices, vertices.shape[1], vertices[:, 0, :], axis=1)

        shape_r, shape_c = vertices.shape[0], vertices.shape[1]

        faces = []
        for i in np.arange(shape_r - 1):
            for j in np.arange(shape_c - 1):
                a_i = [i, j]
                b_i = [i, j + 1]
                c_i = [i + 1, j]
                d_i = [i + 1, j + 1]

                # Avoiding overlapping triangles when tessellating the bottom.
                if not i == 0:
                    faces.append([a_i, b_i, c_i])

                # Avoiding overlapping triangles when tessellating the top.
                if not i == shape_r - 2:
                    faces.append([c_i, b_i, d_i])

        indices = np.ravel_multi_index(
            np.transpose(as_int_array(faces)), [shape_r, shape_c])

        GDB_t = trimesh.Trimesh(
            vertices=vertices.reshape([-1, 3]),
            faces=np.transpose(indices),
            validate=True)

        if not GDB_t.is_watertight:
            warning('Tessellated mesh has holes!')

        return GDB_t


if __name__ == '__main__':
    # 9c91accdd8ea9c39437694bb3265fa6b09fd87d2
    # rd
    # source Environments/plotly/bin/activate
    # export PYTHONPATH=$PYTHONPATH:/Users/kelsolaar/Documents/Development/colour-science/colour:/Users/kelsolaar/Documents/Development/colour-science/trimesh
    # python /Users/kelsolaar/Documents/Development/colour-science/colour/colour/gamut/boundary/common.py

    import trimesh
    import trimesh.smoothing

    import colour
    import colour.plotting
    from colour.plotting import render
    from colour.gamut import gamut_boundary_descriptor_Morovic2000

    np.set_printoptions(
        formatter={'float': '{:0.2f}'.format}, linewidth=2048, suppress=True)

    def create_plane(width=1,
                     height=1,
                     width_segments=1,
                     height_segments=1,
                     direction='+z'):

        x_grid = width_segments
        y_grid = height_segments

        x_grid1 = x_grid + 1
        y_grid1 = y_grid + 1

        # Positions, normals and texcoords.
        positions = np.zeros(x_grid1 * y_grid1 * 3)
        normals = np.zeros(x_grid1 * y_grid1 * 3)
        texcoords = np.zeros(x_grid1 * y_grid1 * 2)

        y = np.arange(y_grid1) * height / y_grid - height / 2
        x = np.arange(x_grid1) * width / x_grid - width / 2

        positions[::3] = np.tile(x, y_grid1)
        positions[1::3] = -np.repeat(y, x_grid1)

        normals[2::3] = 1

        texcoords[::2] = np.tile(np.arange(x_grid1) / x_grid, y_grid1)
        texcoords[1::2] = np.repeat(1 - np.arange(y_grid1) / y_grid, x_grid1)

        # Faces and outline.
        faces, outline = [], []
        for i_y in range(y_grid):
            for i_x in range(x_grid):
                a = i_x + x_grid1 * i_y
                b = i_x + x_grid1 * (i_y + 1)
                c = (i_x + 1) + x_grid1 * (i_y + 1)
                d = (i_x + 1) + x_grid1 * i_y

                faces.extend(((a, b, d), (b, c, d)))
                outline.extend(((a, b), (b, c), (c, d), (d, a)))

        positions = np.reshape(positions, (-1, 3))
        texcoords = np.reshape(texcoords, (-1, 2))
        normals = np.reshape(normals, (-1, 3))

        faces = np.reshape(faces, (-1, 3)).astype(np.uint32)
        outline = np.reshape(outline, (-1, 2)).astype(np.uint32)

        direction = direction.lower()
        if direction in ('-x', '+x'):
            shift, neutral_axis = 1, 0
        elif direction in ('-y', '+y'):
            shift, neutral_axis = -1, 1
        elif direction in ('-z', '+z'):
            shift, neutral_axis = 0, 2

        sign = -1 if '-' in direction else 1

        positions = np.roll(positions, shift, -1)
        normals = np.roll(normals, shift, -1) * sign
        colors = np.ravel(positions)
        colors = np.hstack((np.reshape(
            np.interp(colors, (np.min(colors), np.max(colors)), (0, 1)),
            positions.shape), np.ones((positions.shape[0], 1))))
        colors[..., neutral_axis] = 0

        vertices = np.zeros(
            positions.shape[0],
            [('position', np.float32, 3), ('texcoord', np.float32, 2),
             ('normal', np.float32, 3), ('color', np.float32, 4)])

        vertices['position'] = positions
        vertices['texcoord'] = texcoords
        vertices['normal'] = normals
        vertices['color'] = colors

        return vertices, faces, outline

    def create_box(width=1,
                   height=1,
                   depth=1,
                   width_segments=1,
                   height_segments=1,
                   depth_segments=1,
                   planes=None):
        planes = (('+x', '-x', '+y', '-y', '+z', '-z')
                  if planes is None else [d.lower() for d in planes])

        w_s, h_s, d_s = width_segments, height_segments, depth_segments

        planes_m = []
        if '-z' in planes:
            planes_m.append(create_plane(width, depth, w_s, d_s, '-z'))
            planes_m[-1][0]['position'][..., 2] -= height / 2
        if '+z' in planes:
            planes_m.append(create_plane(width, depth, w_s, d_s, '+z'))
            planes_m[-1][0]['position'][..., 2] += height / 2

        if '-y' in planes:
            planes_m.append(create_plane(height, width, h_s, w_s, '-y'))
            planes_m[-1][0]['position'][..., 1] -= depth / 2
        if '+y' in planes:
            planes_m.append(create_plane(height, width, h_s, w_s, '+y'))
            planes_m[-1][0]['position'][..., 1] += depth / 2

        if '-x' in planes:
            planes_m.append(create_plane(depth, height, d_s, h_s, '-x'))
            planes_m[-1][0]['position'][..., 0] -= width / 2
        if '+x' in planes:
            planes_m.append(create_plane(depth, height, d_s, h_s, '+x'))
            planes_m[-1][0]['position'][..., 0] += width / 2

        positions = np.zeros((0, 3), dtype=np.float32)
        texcoords = np.zeros((0, 2), dtype=np.float32)
        normals = np.zeros((0, 3), dtype=np.float32)

        faces = np.zeros((0, 3), dtype=np.uint32)
        outline = np.zeros((0, 2), dtype=np.uint32)

        offset = 0
        for vertices_p, faces_p, outline_p in planes_m:
            positions = np.vstack((positions, vertices_p['position']))
            texcoords = np.vstack((texcoords, vertices_p['texcoord']))
            normals = np.vstack((normals, vertices_p['normal']))

            faces = np.vstack((faces, faces_p + offset))
            outline = np.vstack((outline, outline_p + offset))
            offset += vertices_p['position'].shape[0]

        vertices = np.zeros(
            positions.shape[0],
            [('position', np.float32, 3), ('texcoord', np.float32, 2),
             ('normal', np.float32, 3), ('color', np.float32, 4)])

        colors = np.ravel(positions)
        colors = np.hstack((np.reshape(
            np.interp(colors, (np.min(colors), np.max(colors)), (0, 1)),
            positions.shape), np.ones((positions.shape[0], 1))))

        vertices['position'] = positions
        vertices['texcoord'] = texcoords
        vertices['normal'] = normals
        vertices['color'] = colors

        return vertices, faces, outline

    m, n = 3, 8
    t = 3
    Hab = np.tile(np.arange(-180, 180, 45) / 360, t)
    C = np.hstack([
        np.ones(int(len(Hab) / t)) * 0.25,
        np.ones(int(len(Hab) / t)) * 0.5,
        np.ones(int(len(Hab) / t)) * 1.0,
    ])
    L = np.hstack([
        np.ones(int(len(Hab) / t)) * 1.0,
        np.ones(int(len(Hab) / t)) * 0.5,
        np.ones(int(len(Hab) / t)) * 0.25,
    ])

    LCHab = tstack([L, C, Hab])
    Jab = colour.convert(
        LCHab, 'CIE LCHab', 'CIE Lab', verbose={'describe': 'short'}) * 100

    np.random.seed(16)
    RGB = np.random.random([200, 200, 3])

    s = 192
    RGB = colour.plotting.geometry.cube(
        width_segments=s, height_segments=s, depth_segments=s)
    Jab_E = colour.convert(
        RGB, 'RGB', 'CIE Lab', verbose={'describe': 'short'}) * 100

    Jab = Jab_E

    vertices, faces, outline = create_box(1, 1, 1, 16, 16, 16)
    RGB_r = vertices['position'] + 0.5
    Jab_r = colour.convert(
        RGB_r, 'RGB', 'CIE Lab', verbose={'describe': 'short'}) * 100
    mesh_r = trimesh.Trimesh(
        vertices=Jab_r.reshape([-1, 3]), faces=faces, validate=True)
    mesh_r.fix_normals()
    # mesh_r.show()

    segments = (32, 36, 40)
    k_r = 32

    GDB_m_s = []
    for k in segments:
        print('^' * 79)
        print('k', k)
        GDB_m = gamut_boundary_descriptor_Morovic2000(Jab, [50, 0, 0], k, 64)

        GDB_m_c = close_gamut_boundary_descriptor(GDB_m, Jab, [50, 0, 0])
        d = GDB_m_c.shape[0] - GDB_m.shape[0]
        print('d', d)

        GDB_m_u = unsparse_gamut_boundary_descriptor(GDB_m_c)
        if k != k_r:
            print('!!! Interpolating !!!')
            GDB_m_i = interpolate_gamut_boundary_descriptor(
                GDB_m_u, k_r + d, 64)
        else:
            GDB_m_i = GDB_m_u

        print('GDB_m', GDB_m.shape)
        print('GDB_m_c', GDB_m_c.shape)
        print('GDB_m_u', GDB_m_u.shape)
        print('GDB_m_i', GDB_m_i.shape)

        print(GDB_m[..., 0][:10, :10])
        print(GDB_m_c[..., 0][:10, :10])
        print(GDB_m_u[..., 0][:10, :10])
        print(GDB_m_i[..., 0][:10, :10])

        # GDB_m_i[GDB_m_i.shape[0] // 2:, ...] += (
        #     np.random.rand(*(GDB_m_i[GDB_m_i.shape[0] // 2:, ...].shape)) * 25)
        GDB_m_s.append(GDB_m_i)

    # print('^' * 79)
    # print(GDB_m_i[..., 0])
    # print(GDB_m[..., 1])
    # print(GDB_m_i[..., 1])
    # print(GDB_m[..., 2])
    # print(GDB_m_i[..., 2])

    import matplotlib.pyplot as plt

    def plot_gamut_boundary_descriptors(GDB_m, columns=None, **kwargs):
        GDB_m = [as_float_array(GDB_m_c) for GDB_m_c in GDB_m]

        assert len(np.unique([GDB_m_c.shape for GDB_m_c in GDB_m])) <= 3, (
            'Gamut boundary descriptor matrices have incompatible shapes!')

        shape_r, shape_c, = GDB_m[0].shape[0], GDB_m[0].shape[1]

        if columns is None:
            columns = shape_c

        figure, axes_a = plt.subplots(
            DEFAULT_INT_DTYPE(np.ceil(shape_c / columns)),
            columns,
            sharex='col',
            sharey='row',
            gridspec_kw={
                'hspace': 0,
                'wspace': 0
            })

        axes_a = np.ravel(axes_a)

        for i in range(shape_c):
            for j in range(len(GDB_m)):
                axes_a[i].plot(
                    GDB_m[j][..., i, 1],
                    orient(GDB_m[j][..., i, 0], 'Flop'),
                    label='{0:d} $^\\degree$'.format(
                        int(i / GDB_m_i.shape[1] * 360)))

                axes_a[i].legend()

        render(**kwargs)

    # plot_gamut_boundary_descriptors(GDB_m_s, 6)

    GDB_t_s = [
        tessellate_gamut_boundary_descriptor(GDB_m) for GDB_m in GDB_m_s
    ]

    # trimesh.smoothing.filter_laplacian(GDB_t, iterations=25)
    # trimesh.repair.broken_faces(GDB_t, color=(255, 0, 0, 255))

    # GDB_t.export('/Users/kelsolaar/Downloads/mesh.dae', 'dae')

    GDB_t_r = GDB_t_s[0]
    # for i, GDB_t in enumerate(GDB_t_s[0:]):
    #     GDB_t.vertices += [(i + 1) * 100, 0, 0]
    #     GDB_t_r = GDB_t_r + GDB_t

    mesh_r.vertices += [50, 0, -150]
    GDB_t_r = GDB_t_r + mesh_r

    trimesh.smoothing.filter_laplacian(GDB_t_r, iterations=25)

    GDB_t_r.show()
