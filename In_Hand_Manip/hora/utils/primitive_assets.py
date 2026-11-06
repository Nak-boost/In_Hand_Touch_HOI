"""Small reproducible URDF primitives; generated files belong in the ignored cache."""
import math
from pathlib import Path


def cylinder_asset(directory, size, density=500.0):
    diameter, second_diameter, height = map(float, size)
    if not all(math.isfinite(v) and v > 0 for v in size) or abs(diameter-second_diameter) > 1e-9:
        raise ValueError('Cylinder size must be [diameter, diameter, height] in metres')
    radius = diameter / 2
    mass = density * math.pi * radius**2 * height
    transverse = mass * (3*radius**2 + height**2) / 12
    axial = mass * radius**2 / 2
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Preview 4's importer treats the first dot as the extension delimiter.
    path = directory / ('cylinder_d%d_h%d.urdf' % (round(diameter*1e9), round(height*1e9)))
    geometry = '<geometry><cylinder radius="%.12g" length="%.12g"/></geometry>' % (radius, height)
    content = ('<robot name="cylinder"><link name="object">'
               '<inertial><mass value="%.12g"/><inertia ixx="%.12g" iyy="%.12g" izz="%.12g" '
               'ixy="0" ixz="0" iyz="0"/></inertial>'
               '<visual>%s<material name="object"><color rgba="0.8 0.5 0.15 1"/></material></visual>'
               '<collision>%s</collision></link></robot>\n') % (mass, transverse, transverse, axial, geometry, geometry)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
    return path
