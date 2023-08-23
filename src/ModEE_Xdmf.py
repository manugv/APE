#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  13 13:07:49 2021.

@author: Manu Goudar
"""

# Module that has functions to write data
import xml.etree.ElementTree as ET
from xml.dom import minidom


def createdataitem(
    parentnode, Name, Dimensions, NumberType, Precision, Format, datafile, ItemType=None, Seek=None
):
    """
    Name         (no default)
    ItemType     Uniform|Collection|tree|HyperSlab|coordinates|Function
    Dimensions   (no default) in KJI Order
    NumberType   Float | Int | UInt | Char | UChar
    Precision    1 | 2 (Int or UInt only) |4 | 8
    Format       XML | HDF | Binary
    Endian       Native | Big | Little (applicable only to Binary format)
    Compression  Raw|Zlib|BZip2 (applicable only to Binary format, and
                 depend on xdmf configuration))
    Seek         0 (number of bytes to skip, applicable only to Binary
                 format with Raw compression)
    """
    # Write data to xml
    node = ET.SubElement(parentnode, "DataItem")
    if Name is not None:
        node.set("Name", Name)
    node.set("NumberType", NumberType)
    node.set("Dimensions", Dimensions)
    node.set("Precision", Precision)
    node.set("Format", Format)
    if Seek is not None:
        node.set("Seek", Seek)
    node.text = datafile


def createattribute(parentnode, Name, att_type, center):
    """
    ELEMENT Attribute (Information*, DataItem*)
    Name CDATA
    Center (Node | Cell | Grid | Face | Edge | Other) "Node"
    AttributeType (Scalar | Vector | Tensor | Tensor6 | Matrix) "Scalar"
    ItemType (FiniteElementFunction) #IMPLIED
    ElementFamily (CG | DG | RT | BDM | CR | N1curl | N2curl) #IMPLIED
    ElementDegree CDATA #IMPLIED
    ElementCell CDATA #IMPLIED
    """
    # Write an attribute in xml
    attribute = ET.SubElement(parentnode, "Attribute")
    attribute.set("Name", Name)  # Set name
    attribute.set("AttributeType", att_type)
    attribute.set("Center", center)  # Set center of the attribute
    return attribute


def createtopology(
    parentnode, name, topologytype, Dimensions=None, NodesPerElement=None, Order=None
):
    """
    Name
    TopologyType (NoTopologyType | Polyvertex | 2DSMesh | 2DRectMesh |
                 2DCoRectMesh | 3DSMesh | 3DRectMesh | 3DCoRectMesh) |
                 Polyvertex | Polyline | Polygon | Triangle |
                 Quadrilateral | Tetrahedron | Pyramid| Wedge |
                 Hexahedron | Edge_3 | Triangle_6 | Quadrilateral_8 |
                 Tetrahedron_10 | Pyramid_13 | Mixed | Wedge_15
    Dimensions
    NodesPerElement (no default) Only Important for
                    Polyvertex, Polygon and Polyline
    Order           each cell type has its own default
    """
    topo = ET.SubElement(parentnode, "Topology")
    if name is not None:
        topo.set("Name", name)  # Set name
    topo.set("TopologyType", topologytype)  # Set center of the attribute
    if topologytype == "Polyvertex":
        topo.set("NodesPerElement", NodesPerElement)
    else:
        topo.set("Dimensions", Dimensions)


def creategeometry(parentnode, name, geotype):
    """
    Name
    GeometryType (XYZ | XY | X_Y_Z | VXVYVZ | ORIGIN_DXDYDZ) "XYZ"
    """
    geo = ET.SubElement(parentnode, "Geometry")
    if name is not None:
        geo.set("Name", name)  # Set name
    geo.set("GeometryType", geotype)  # X_Y_Z
    return geo


def createdomain(parentnode):
    return ET.SubElement(parentnode, "Domain")


def creategrid(parentnode, Name, GridType, CollectionType):
    """
    Name            (no default)
    GridType         Uniform | Collection | Tree | Subset
    CollectionType   Spatial | Temporal (Only Meaningful
                                         GridType="Collection")
    """
    grid = ET.SubElement(parentnode, "Grid")
    grid.set("Name", Name)  # Set name
    grid.set("GridType", GridType)
    if CollectionType is not None:
        grid.set("CollectionType", CollectionType)
    return grid


def createtimevalue(parentnode, value):
    tt = ET.SubElement(parentnode, "Time")
    tt.set("Value", value)


def createheader():
    data = ET.Element("Xdmf")
    data.set("xmlns:xi", "http://www.w3.org/2001/XInclude")
    data.set("Version", "2.0")
    return data


def createreference(parentnode, name, attrib):
    ref = "/Xdmf/Domain/" + attrib
    tt = ET.SubElement(parentnode, name)
    tt.set("Reference", ref)


def writexml(root, filename):
    """
    Print pretty XML
    """
    rough_string = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(rough_string)
    myfile = open(filename + ".xmf", "w")
    reparsed.writexml(myfile, addindent="  ", newl="\n")
    myfile.close()
