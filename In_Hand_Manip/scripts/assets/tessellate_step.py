"""Tessellate official STEP geometry without changing units or assembling parts.

Offline tooling only: Python 3.12 + cadquery-ocp + numpy + trimesh.
STEPControl converts source length units to millimetres; exported OBJ uses metres.
"""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location


def mesh_shape(shape, tolerance_mm):
    BRepMesh_IncrementalMesh(shape, tolerance_mm, False, .1, True).Perform()
    vertices, triangles = [], []
    faces=TopExp_Explorer(shape,TopAbs_FACE)
    while faces.More():
        face=TopoDS.Face(faces.Current());location=TopLoc_Location()
        tri=BRep_Tool.Triangulation_s(face,location)
        if tri is not None:
            offset=len(vertices)
            for i in range(1,tri.NbNodes()+1):
                p=tri.Node(i).Transformed(location.Transformation())
                vertices.append([p.X()*.001,p.Y()*.001,p.Z()*.001])
            for i in range(1,tri.NbTriangles()+1):
                a,b,c=tri.Triangle(i).Get()
                if face.Orientation()==TopAbs_REVERSED:b,c=c,b
                triangles.append([offset+a-1,offset+b-1,offset+c-1])
        faces.Next()
    return trimesh.Trimesh(vertices=vertices,faces=triangles,process=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('--tolerance-mm',type=float,default=.02)
    args=parser.parse_args()
    reader=STEPControl_Reader()
    if reader.ReadFile(str(args.source))!=IFSelect_RetDone:raise ValueError('STEP read failed')
    reader.TransferRoots();shape=reader.OneShape()
    solids=TopExp_Explorer(shape,TopAbs_SOLID)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index=0
    while solids.More():
        mesh=mesh_shape(solids.Current(),args.tolerance_mm)
        dest=args.output_dir/(args.source.stem+'_'+str(index)+'.obj')
        mesh.export(dest)
        print(dest, 'bounds_m',mesh.bounds.tolist(),'faces',len(mesh.faces),'watertight',mesh.is_watertight)
        solids.Next();index+=1


if __name__=='__main__':main()
