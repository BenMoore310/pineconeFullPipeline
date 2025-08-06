#!/usr/bin/env python


import time
import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'/home/bm424/Projects/Catmull-Clark')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS

startTime = time.time()

geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
O_1 = geompy.MakeVertex(0, 0, 0)
OX_1 = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY_1 = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ_1 = geompy.MakeVectorDXDYDZ(0, 0, 1)
Vertex_1 = geompy.MakeVertex(-5, 0, 0)
Circle_1 = geompy.MakeCircle(Vertex_1, OY_1, 24)
Face_1 = geompy.MakeFaceWires([Circle_1], 1)
Extrusion_1 = geompy.MakePrismVecH(Face_1, OY_1, 25)
Extrusion_2 = geompy.MakePrismVecH(Face_1, OY_1, -84.31399999999999)
Vertex_0 = geompy.MakeVertex(0, 9.061, 0)
Circle_2 = geompy.MakeCircle(Vertex_0, OX_1, 6)
Face_2 = geompy.MakeFaceWires([Circle_2], 1)
Extrusion_3 = geompy.MakePrismVecH(Face_2, OX_1, -87)
Extrusion_4 = geompy.MakePrismVecH(Face_2, OX_1, 60)
Vertex_2 = geompy.MakeVertex(6.991, 25, 0)
Vertex_3 = geompy.MakeVertex(6.741, 25, 0)
Vertex_4 = geompy.MakeVertex(6.991, 0, 0)
Vertex_5 = geompy.MakeVertex(6.741, 0, 0)
Vertex_6 = geompy.MakeVertex(6.02, -0.512, 0)
Vertex_7 = geompy.MakeVertex(6.596, -0.512, 0)
Vertex_8 = geompy.MakeVertex(6.02, -48.423, 0)
Vertex_9 = geompy.MakeVertex(6.596, -48.999, 0)
Vertex_10 = geompy.MakeVertex(0, -48.423, 0)
Vertex_11 = geompy.MakeVertex(0, -48.999, 0)


# Load the text file containing coordinates
upper_spline = "/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/spline.txt"  
with open(upper_spline, 'r') as file:
    upper_lines = file.readlines()

lower_spline = "/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/offset_spline.txt"  
with open(lower_spline, 'r') as file:
    lower_lines = file.readlines()

# Create vertices from coordinates
upper_vertices = []
lower_vertices = []
i = 1
for line in upper_lines:
    coords = line.strip().split()  # Splitting by whitespace
    x = float(coords[0])
    y = float(coords[1])
    vertex = geompy.MakeVertex(x, y, 0)  # Create a vertex at (x, y, 0)
    upper_vertices.append(vertex)
    geompy.addToStudy( vertex, f"upperVertex_{i}" )
    i += 1

for line in lower_lines:
    coords = line.strip().split()  # Splitting by whitespace
    x = float(coords[0])
    y = float(coords[1])
    vertex = geompy.MakeVertex(x, y, 0)  # Create a vertex at (x, y, 0)
    lower_vertices.append(vertex)
    geompy.addToStudy( vertex, f"lowerVertex_{i}" )
    i += 1


edges = []
for i in range(len(upper_vertices) - 1):
    edge = geompy.MakeEdge(upper_vertices[i], upper_vertices[i+1])
    edges.append(edge)

upper_wire = geompy.MakeWire(edges)
geompy.addToStudy(upper_wire, 'upper_wire')

revolution_upper_1 = geompy.MakeRevolution(upper_wire, OY, 360*math.pi/180.0)
geompy.addToStudy(revolution_upper_1, 'revolution_upper')

Translation_upper_1 = geompy.MakeTranslation(revolution_upper_1, 0, -3, 0)
Translation_upper_2 = geompy.MakeTranslation(Translation_upper_1, 0, -3, 0)
Translation_upper_3 = geompy.MakeTranslation(Translation_upper_2, 0, -3, 0)
Translation_upper_4 = geompy.MakeTranslation(Translation_upper_3, 0, -3, 0)
Translation_upper_5 = geompy.MakeTranslation(Translation_upper_4, 0, -3, 0)
Translation_upper_6 = geompy.MakeTranslation(Translation_upper_5, 0, -3, 0)
Translation_upper_7 = geompy.MakeTranslation(Translation_upper_6, 0, -3, 0)
Translation_upper_8 = geompy.MakeTranslation(Translation_upper_7, 0, -3, 0)
Translation_upper_9 = geompy.MakeTranslation(Translation_upper_8, 0, -3, 0)
Translation_upper_10 = geompy.MakeTranslation(Translation_upper_9, 0, -3, 0)
Translation_upper_11= geompy.MakeTranslation(Translation_upper_10, 0, -3, 0)
Translation_upper_12= geompy.MakeTranslation(Translation_upper_11, 0, -3, 0)
Translation_upper_13= geompy.MakeTranslation(Translation_upper_12, 0, -3, 0)
Translation_upper_14= geompy.MakeTranslation(Translation_upper_13, 0, -3, 0)
Translation_upper_15= geompy.MakeTranslation(Translation_upper_14, 0, -3, 0)



geompy.addToStudy(Translation_upper_1, 'Translation_upper_1')
geompy.addToStudy(Translation_upper_2, 'Translation_upper_2')
geompy.addToStudy(Translation_upper_3, 'Translation_upper_3')
geompy.addToStudy(Translation_upper_4, 'Translation_upper_4')
geompy.addToStudy(Translation_upper_5, 'Translation_upper_5')
geompy.addToStudy(Translation_upper_6, 'Translation_upper_6')
geompy.addToStudy(Translation_upper_7, 'Translation_upper_7')
geompy.addToStudy(Translation_upper_8, 'Translation_upper_8')
geompy.addToStudy(Translation_upper_9, 'Translation_upper_9')
geompy.addToStudy(Translation_upper_10, 'Translation_upper_10')
geompy.addToStudy(Translation_upper_11, 'Translation_upper_11')
geompy.addToStudy(Translation_upper_12, 'Translation_upper_12')
geompy.addToStudy(Translation_upper_13, 'Translation_upper_13')
geompy.addToStudy(Translation_upper_14, 'Translation_upper_14')
geompy.addToStudy(Translation_upper_15, 'Translation_upper_15')


Fuse_upper = geompy.MakeFuseList([revolution_upper_1, Translation_upper_1, Translation_upper_2, Translation_upper_3, Translation_upper_4,
                                  Translation_upper_5, Translation_upper_6, Translation_upper_7, Translation_upper_8, Translation_upper_9,
                                  Translation_upper_10, Translation_upper_11, Translation_upper_12, Translation_upper_13, Translation_upper_14,
                                  Translation_upper_15], False, False)

geompy.addToStudy(Fuse_upper, 'Fuse_upper')


edge = geompy.MakeEdge(upper_vertices[0], lower_vertices[-1])
edges.append(edge)

lowerEdges = []
for i in range(len(lower_vertices) - 1):
    edge = geompy.MakeEdge(lower_vertices[i], lower_vertices[i+1])
    edges.append(edge)
    lowerEdges.append(edge)
# edge = geompy.MakeEdge(vertices[-1], vertices[0])

####################################################################################

lowerWire = geompy.MakeWire(lowerEdges)
# geompy.addToStudy(upper_wire, 'upper_wire')

revolution_lower_1 = geompy.MakeRevolution(lowerWire, OY, 360*math.pi/180.0)
geompy.addToStudy(revolution_lower_1, 'revolution_lower')

Translation_lower_1 = geompy.MakeTranslation(revolution_lower_1, 0, -3, 0)
Translation_lower_2 = geompy.MakeTranslation(Translation_lower_1, 0, -3, 0)
Translation_lower_3 = geompy.MakeTranslation(Translation_lower_2, 0, -3, 0)
Translation_lower_4 = geompy.MakeTranslation(Translation_lower_3, 0, -3, 0)
Translation_lower_5 = geompy.MakeTranslation(Translation_lower_4, 0, -3, 0)
Translation_lower_6 = geompy.MakeTranslation(Translation_lower_5, 0, -3, 0)
Translation_lower_7 = geompy.MakeTranslation(Translation_lower_6, 0, -3, 0)
Translation_lower_8 = geompy.MakeTranslation(Translation_lower_7, 0, -3, 0)
Translation_lower_9 = geompy.MakeTranslation(Translation_lower_8, 0, -3, 0)
Translation_lower_10 = geompy.MakeTranslation(Translation_lower_9, 0, -3, 0)
Translation_lower_11= geompy.MakeTranslation(Translation_lower_10, 0, -3, 0)
Translation_lower_12= geompy.MakeTranslation(Translation_lower_11, 0, -3, 0)
Translation_lower_13= geompy.MakeTranslation(Translation_lower_12, 0, -3, 0)
Translation_lower_14= geompy.MakeTranslation(Translation_lower_13, 0, -3, 0)
Translation_lower_15= geompy.MakeTranslation(Translation_lower_14, 0, -3, 0)

Fuse_lower = geompy.MakeFuseList([revolution_lower_1, Translation_lower_1, Translation_lower_2, Translation_lower_3, Translation_lower_4,
                                  Translation_lower_5, Translation_lower_6, Translation_lower_7, Translation_lower_8, Translation_lower_9,
                                  Translation_lower_10, Translation_lower_11, Translation_lower_12, Translation_lower_13, Translation_lower_14,
                                  Translation_lower_15], False, False)

geompy.addToStudy(Fuse_lower, 'Fuse_lower')

geompy.ExportSTL(Fuse_lower, "/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/Fuse_lower.stl", True, 0.001, True)

####################################################################################

#vertices and lines taken from original pinecone case - form walls of stack connecting to plates
Vertex_36 = geompy.MakeVertex(7.109, -3.999, 0)
Vertex_37 = geompy.MakeVertex(6.596, -3.999, 0)
Vertex_38 = geompy.MakeVertex(6.596, -4.249, 0)
Vertex_40 = geompy.MakeVertex(6.596, -1.3, 0)

Line_36 = geompy.MakeLineTwoPnt(lower_vertices[0], Vertex_36)
Line_37 = geompy.MakeLineTwoPnt(Vertex_36, Vertex_37)
Line_38 = geompy.MakeLineTwoPnt(Vertex_37, Vertex_38)
Line_33 = geompy.MakeLineTwoPnt(Vertex_40, upper_vertices[-1])

Line_9899 = geompy.MakeLineTwoPnt(lower_vertices[0], lower_vertices[1])

geompy.addToStudy(Line_9899, 'Line_9899')



hood_wire = geompy.MakeWire([Line_38, Line_37, Line_36, Line_9899])

geompy.addToStudy(hood_wire, 'hood_wire')


Revolution_hood = geompy.MakeRevolution(hood_wire, OY, 360*math.pi/180.0)

geompy.addToStudy(Revolution_hood, 'Revolution_hood')

Translation_hood_1 = geompy.MakeTranslation(Revolution_hood, 0, -3, 0)
Translation_hood_2 = geompy.MakeTranslation(Translation_hood_1, 0, -3, 0)
Translation_hood_3 = geompy.MakeTranslation(Translation_hood_2, 0, -3, 0)
Translation_hood_4 = geompy.MakeTranslation(Translation_hood_3, 0, -3, 0)
Translation_hood_5 = geompy.MakeTranslation(Translation_hood_4, 0, -3, 0)
Translation_hood_6 = geompy.MakeTranslation(Translation_hood_5, 0, -3, 0)
Translation_hood_7 = geompy.MakeTranslation(Translation_hood_6, 0, -3, 0)
Translation_hood_8 = geompy.MakeTranslation(Translation_hood_7, 0, -3, 0)
Translation_hood_9 = geompy.MakeTranslation(Translation_hood_8, 0, -3, 0)
Translation_hood_10 = geompy.MakeTranslation(Translation_hood_9, 0, -3, 0)
Translation_hood_11= geompy.MakeTranslation(Translation_hood_10, 0, -3, 0)
Translation_hood_12= geompy.MakeTranslation(Translation_hood_11, 0, -3, 0)
Translation_hood_13= geompy.MakeTranslation(Translation_hood_12, 0, -3, 0)
Translation_hood_14= geompy.MakeTranslation(Translation_hood_13, 0, -3, 0)
Translation_hood_15= geompy.MakeTranslation(Translation_hood_14, 0, -3, 0)

fullHood = geompy.MakeFuseList([Revolution_hood, Translation_hood_1, Translation_hood_2, Translation_hood_3, Translation_hood_4,
                                  Translation_hood_5, Translation_hood_6, Translation_hood_7, Translation_hood_8, Translation_hood_9,
                                  Translation_hood_10, Translation_hood_11, Translation_hood_12, Translation_hood_13, Translation_hood_14,
                                  Translation_hood_15], True, True)

geompy.addToStudy(fullHood, 'fullHood')


# edges.append(edge)

plate_wire = geompy.MakeWire(edges)

# geompy.addToStudy(plate_wire, 'plate_wire')

# wire = geompy.MakeWire([Line_33, plate_wire])

# Create wire from edges
one_plate_wire = geompy.MakeWire([Line_33, plate_wire, Line_36, Line_37, Line_38], 1e-07)
one_plate_wire_reduced = geompy.MakeWire([Line_33, plate_wire, Line_36, Line_37], 1e-07)
geompy.addToStudy(one_plate_wire_reduced, 'one_plate_wire_reduced')
geompy.addToStudy(one_plate_wire, 'wire_1')

wire_1_vertex_bottom = geompy.SubShape(one_plate_wire, geompy.ShapeType["VERTEX"], [0]) 
geompy.addToStudy(wire_1_vertex_bottom, 'wire_1_bottom')

wire_1_vertex_top = geompy.SubShape(one_plate_wire, geompy.ShapeType["VERTEX"], [1]) 
geompy.addToStudy(wire_1_vertex_top, 'wire_1_top')



Translation_1 = geompy.MakeTranslation(one_plate_wire, 0, -3, 0)

translation_1_vertex_top = geompy.SubShape(Translation_1, geompy.ShapeType["VERTEX"], [1])
translation_1_vertex_bottom = geompy.SubShape(Translation_1, geompy.ShapeType["VERTEX"], [0])
# geompy.addToStudy(translation_1_vertex_top, 'translation_1_vertex_top')

line_1a = geompy.MakeLineTwoPnt(wire_1_vertex_bottom, translation_1_vertex_top)
geompy.addToStudy(line_1a, 'line_1a')

Translation_2 = geompy.MakeTranslation(Translation_1, 0, -3, 0)

translation_2_vertex_top = geompy.SubShape(Translation_2, geompy.ShapeType["VERTEX"], [1])
translation_2_vertex_bottom = geompy.SubShape(Translation_2, geompy.ShapeType["VERTEX"], [0])
line_2a = geompy.MakeLineTwoPnt(translation_1_vertex_bottom, translation_2_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')



Translation_3 = geompy.MakeTranslation(Translation_2, 0, -3, 0)

translation_3_vertex_top = geompy.SubShape(Translation_3, geompy.ShapeType["VERTEX"], [1])
translation_3_vertex_bottom = geompy.SubShape(Translation_3, geompy.ShapeType["VERTEX"], [0])
line_3a = geompy.MakeLineTwoPnt(translation_2_vertex_bottom, translation_3_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_4 = geompy.MakeTranslation(Translation_3, 0, -3, 0)

translation_4_vertex_top = geompy.SubShape(Translation_4, geompy.ShapeType["VERTEX"], [1])
translation_4_vertex_bottom = geompy.SubShape(Translation_4, geompy.ShapeType["VERTEX"], [0])
line_4a = geompy.MakeLineTwoPnt(translation_3_vertex_bottom, translation_4_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')


Translation_5 = geompy.MakeTranslation(Translation_4, 0, -3, 0)

translation_5_vertex_top = geompy.SubShape(Translation_5, geompy.ShapeType["VERTEX"], [1])
translation_5_vertex_bottom = geompy.SubShape(Translation_5, geompy.ShapeType["VERTEX"], [0])
line_5a = geompy.MakeLineTwoPnt(translation_4_vertex_bottom, translation_5_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_6 = geompy.MakeTranslation(Translation_5, 0, -3, 0)

translation_6_vertex_top = geompy.SubShape(Translation_6, geompy.ShapeType["VERTEX"], [1])
translation_6_vertex_bottom = geompy.SubShape(Translation_6, geompy.ShapeType["VERTEX"], [0])
line_6a = geompy.MakeLineTwoPnt(translation_5_vertex_bottom, translation_6_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_7 = geompy.MakeTranslation(Translation_6, 0, -3, 0)

translation_7_vertex_top = geompy.SubShape(Translation_7, geompy.ShapeType["VERTEX"], [1])
translation_7_vertex_bottom = geompy.SubShape(Translation_7, geompy.ShapeType["VERTEX"], [0])
line_7a = geompy.MakeLineTwoPnt(translation_6_vertex_bottom, translation_7_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_8 = geompy.MakeTranslation(Translation_7, 0, -3, 0)

translation_8_vertex_top = geompy.SubShape(Translation_8, geompy.ShapeType["VERTEX"], [1])
translation_8_vertex_bottom = geompy.SubShape(Translation_8, geompy.ShapeType["VERTEX"], [0])
line_8a = geompy.MakeLineTwoPnt(translation_7_vertex_bottom, translation_8_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_9 = geompy.MakeTranslation(Translation_8, 0, -3, 0)

translation_9_vertex_top = geompy.SubShape(Translation_9, geompy.ShapeType["VERTEX"], [1])
translation_9_vertex_bottom = geompy.SubShape(Translation_9, geompy.ShapeType["VERTEX"], [0])
line_9a = geompy.MakeLineTwoPnt(translation_8_vertex_bottom, translation_9_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_10 = geompy.MakeTranslation(Translation_9, 0, -3, 0)

translation_10_vertex_top = geompy.SubShape(Translation_10, geompy.ShapeType["VERTEX"], [1])
translation_10_vertex_bottom = geompy.SubShape(Translation_10, geompy.ShapeType["VERTEX"], [0])
line_10a = geompy.MakeLineTwoPnt(translation_9_vertex_bottom, translation_10_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_11 = geompy.MakeTranslation(Translation_10, 0, -3, 0)

translation_11_vertex_top = geompy.SubShape(Translation_11, geompy.ShapeType["VERTEX"], [1])
translation_11_vertex_bottom = geompy.SubShape(Translation_11, geompy.ShapeType["VERTEX"], [0])
line_11a = geompy.MakeLineTwoPnt(translation_10_vertex_bottom, translation_11_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_12 = geompy.MakeTranslation(Translation_11, 0, -3, 0)

translation_12_vertex_top = geompy.SubShape(Translation_12, geompy.ShapeType["VERTEX"], [1])
translation_12_vertex_bottom = geompy.SubShape(Translation_12, geompy.ShapeType["VERTEX"], [0])
line_12a = geompy.MakeLineTwoPnt(translation_11_vertex_bottom, translation_12_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_13 = geompy.MakeTranslation(Translation_12, 0, -3, 0)

translation_13_vertex_top = geompy.SubShape(Translation_13, geompy.ShapeType["VERTEX"], [1])
translation_13_vertex_bottom = geompy.SubShape(Translation_13, geompy.ShapeType["VERTEX"], [0])
line_13a = geompy.MakeLineTwoPnt(translation_12_vertex_bottom, translation_13_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_14 = geompy.MakeTranslation(Translation_13, 0, -3, 0)

translation_14_vertex_top = geompy.SubShape(Translation_14, geompy.ShapeType["VERTEX"], [1])
translation_14_vertex_bottom = geompy.SubShape(Translation_14, geompy.ShapeType["VERTEX"], [0])
line_14a = geompy.MakeLineTwoPnt(translation_13_vertex_bottom, translation_14_vertex_top)
# geompy.addToStudy(line_2a, 'line_1a')

Translation_15 = geompy.MakeTranslation(one_plate_wire_reduced, 0, -45, 0)

translation_15_vertex_top = geompy.SubShape(Translation_15, geompy.ShapeType["VERTEX"], [1])
translation_15_vertex_bottom = geompy.SubShape(Translation_15, geompy.ShapeType["VERTEX"], [0])
line_15a = geompy.MakeLineTwoPnt(translation_14_vertex_bottom, translation_15_vertex_top)

geompy.addToStudy(Translation_1, 'translation')

#add lines connecting vertices making up the back of the stack

Line_16a = geompy.MakeLineTwoPnt(Vertex_11, translation_15_vertex_bottom)
Line_17a = geompy.MakeLineTwoPnt(Vertex_11, Vertex_10)
Line_18a = geompy.MakeLineTwoPnt(Vertex_10, Vertex_8)
Line_19a = geompy.MakeLineTwoPnt(Vertex_8, Vertex_6)
Line_20a = geompy.MakeLineTwoPnt(Vertex_6, Vertex_5)
Line_21a = geompy.MakeLineTwoPnt(Vertex_5, Vertex_3)
Line_22a = geompy.MakeLineTwoPnt(Vertex_3, Vertex_2)
Line_23a = geompy.MakeLineTwoPnt(Vertex_2, Vertex_4)
Line_24a = geompy.MakeLineTwoPnt(Vertex_4, Vertex_7)
Line_25a = geompy.MakeLineTwoPnt(Vertex_7, wire_1_vertex_top)


wire_big = geompy.MakeWire([one_plate_wire, Translation_1, Translation_2, Translation_3, Translation_4, Translation_5, Translation_6, Translation_7, Translation_8, Translation_9
                            , Translation_10, Translation_11, Translation_12, Translation_13, Translation_14, Translation_15, line_1a, line_2a, line_3a, line_4a, line_5a, line_6a
                             , line_7a, line_8a, line_9a, line_10a, line_11a, line_12a, line_13a, line_14a, line_15a, Line_16a, Line_17a, Line_18a, Line_19a, Line_20a, Line_21a
                             , Line_22a, Line_23a, Line_24a, Line_25a ])


geompy.addToStudy(wire_big, 'wire_big')

Fillet_1D_2 = geompy.MakeFillet1D(wire_big, 0.75, [6, 8, 10, 16, 18])

geompy.addToStudy(Fillet_1D_2, 'fillet_1d_2')

Face_3 = geompy.MakeFaceWires([Fillet_1D_2], 1)
geompy.addToStudy(Face_3, 'face_3')
Revolution_1 = geompy.MakeRevolution(Face_3, OY, 360*math.pi/180.0)

geompy.addToStudy(Revolution_1, 'Revolution_1')

print('made revolution 1')

Vertex_43 = geompy.MakeVertex(-1.063, -3.999, 0)
Vertex_44 = geompy.MakeVertex(1.063, -3.999, 0)
Vertex_47 = geompy.MakeVertex(-1.015, -5.399, 0)
Vertex_48 = geompy.MakeVertex(1.015, -5.399, 0)
Line_5 = geompy.MakeLineTwoPnt(Vertex_43, Vertex_44)
Line_6 = geompy.MakeLineTwoPnt(Vertex_44, Vertex_48)
Line_7 = geompy.MakeLineTwoPnt(Vertex_48, Vertex_47)
Line_8 = geompy.MakeLineTwoPnt(Vertex_47, Vertex_43)
Wire_6 = geompy.MakeWire([Line_5, Line_6, Line_7, Line_8], 1e-07)
Face_12 = geompy.MakeFaceWires([Wire_6], 1)
Extrusion_6 = geompy.MakePrismVecH2Ways(Face_12, OZ_1, 6.7)
Rotation_1 = geompy.MakeRotation(Extrusion_6, OY_1, 90*math.pi/180.0)
Fuse_6 = geompy.MakeFuseList([Extrusion_6, Rotation_1], True, True)
Translation_16 = geompy.MakeTranslation(Fuse_6, 0, -3, 0)
Translation_17 = geompy.MakeTranslation(Translation_16, 0, -3, 0)
Translation_18 = geompy.MakeTranslation(Translation_17, 0, -3, 0)
Translation_19 = geompy.MakeTranslation(Translation_18, 0, -3, 0)
Translation_20 = geompy.MakeTranslation(Translation_19, 0, -3, 0)
Translation_21 = geompy.MakeTranslation(Translation_20, 0, -3, 0)
Translation_22 = geompy.MakeTranslation(Translation_21, 0, -3, 0)
Translation_23 = geompy.MakeTranslation(Translation_22, 0, -3, 0)
Translation_24 = geompy.MakeTranslation(Translation_23, 0, -3, 0)
Translation_25 = geompy.MakeTranslation(Translation_24, 0, -3, 0)
Translation_26 = geompy.MakeTranslation(Translation_25, 0, -3, 0)
Translation_27 = geompy.MakeTranslation(Translation_26, 0, -3, 0)
Translation_28 = geompy.MakeTranslation(Translation_27, 0, -3, 0)
Translation_29 = geompy.MakeTranslation(Translation_28, 0, -3, 0)
Vertex_31 = geompy.MakeVertex(1.25, 12.061, 10)
Vertex_32 = geompy.MakeVertex(10, 50, -10)
Box_1 = geompy.MakeBoxTwoPnt(Vertex_31, Vertex_32)
Cut_3 = geompy.MakeCutList(Revolution_1, [Box_1, Fuse_6, Translation_16, Translation_17, Translation_18, Translation_19, Translation_20, Translation_21, Translation_22, Translation_23, Translation_24, Translation_25, Translation_26, Translation_27, Translation_28, Translation_29], True)
Vertex_14 = geompy.MakeVertex(0, 25, 6.741)
Vertex_15 = geompy.MakeVertex(0, 1.311, 6.741)
Vertex_24 = geompy.MakeVertex(0, 25, -6.741)
Vertex_25 = geompy.MakeVertex(0, 1.311, -6.741)
Vertex_26 = geompy.MakeVertex(15.8, 25, 12.05)
Vertex_27 = geompy.MakeVertex(15.8, 1.311, 12.05)
Vertex_28 = geompy.MakeVertex(15.8, 1.311, -12.05)
Vertex_29 = geompy.MakeVertex(15.8, 25, -12.05)
Line_19 = geompy.MakeLineTwoPnt(Vertex_14, Vertex_26)
Line_20 = geompy.MakeLineTwoPnt(Vertex_26, Vertex_27)
Line_21 = geompy.MakeLineTwoPnt(Vertex_27, Vertex_15)
Line_22 = geompy.MakeLineTwoPnt(Vertex_15, Vertex_14)
Line_23 = geompy.MakeLineTwoPnt(Vertex_24, Vertex_25)
Line_24 = geompy.MakeLineTwoPnt(Vertex_25, Vertex_28)
Line_25 = geompy.MakeLineTwoPnt(Vertex_28, Vertex_29)
Line_26 = geompy.MakeLineTwoPnt(Vertex_29, Vertex_24)
Line_27 = geompy.MakeLineTwoPnt(Vertex_25, Vertex_15)
Line_28 = geompy.MakeLineTwoPnt(Vertex_28, Vertex_27)
Line_29 = geompy.MakeLineTwoPnt(Vertex_29, Vertex_26)
Face_6 = geompy.MakeFaceWires([Line_23, Line_24, Line_25, Line_26], 1)
Face_7 = geompy.MakeFaceWires([Line_19, Line_20, Line_21, Line_22], 1)
Face_8 = geompy.MakeFaceWires([Line_20, Line_25, Line_28, Line_29], 1)
Face_9 = geompy.MakeFaceWires([Line_21, Line_24, Line_27, Line_28], 1)
Fuse_1 = geompy.MakeFuseList([Face_6, Face_7, Face_8, Face_9], True, True)
Thickness_1 = geompy.MakeThickSolid(Face_6, 0.25, [], True)
Thickness_2 = geompy.MakeThickSolid(Face_9, 0.25, [], True)
Thickness_3 = geompy.MakeThickSolid(Face_7, 0.25, [], True)
Vertex_30 = geompy.MakeVertex(20, 1.311, 12.05)
Vertex_33 = geompy.MakeVertex(20, 1.311, -12.05)
Line_30 = geompy.MakeLineTwoPnt(Vertex_30, Vertex_33)
Line_31 = geompy.MakeLineTwoPnt(Vertex_27, Vertex_30)
Line_32 = geompy.MakeLineTwoPnt(Vertex_28, Vertex_33)
Face_10 = geompy.MakeFaceWires([Line_28, Line_30, Line_31, Line_32], 1)
Thickness_4 = geompy.MakeThickSolid(Face_10, 0.25, [])
Circle_3 = geompy.MakeCircle(O_1, OY_1, 6.741)
Face_11 = geompy.MakeFaceWires([Circle_3], 1)
Extrusion_5 = geompy.MakePrismVecH(Face_11, OY_1, 15)
Cut_1 = geompy.MakeCutList(Thickness_2, [Extrusion_5], True)
Vertex_42a = geompy.MakeVertex(0, 25, 6.878341)
Vertex_45a = geompy.MakeVertex(1.25, 25, 7.3)
Vertex_46a = geompy.MakeVertex(1.25, 25, 6.75)
Line_39a = geompy.MakeLineTwoPnt(Vertex_45a, Vertex_42a)
Line_40a = geompy.MakeLineTwoPnt(Vertex_46a, Vertex_45a)
Line_41a = geompy.MakeLineTwoPnt(Vertex_42a, Vertex_46a)
Face_13a = geompy.MakeFaceWires([Line_39a, Line_40a, Line_41a], 1)
Extrusion_7a = geompy.MakePrismVecH(Face_13a, OY_1, -23.564)
Plane_1a = geompy.MakePlane(O_1, OZ_1, 50)
Mirror_1a = geompy.MakeMirrorByPlane(Extrusion_7a, Plane_1a)
Vertex_41a = geompy.MakeVertex(-10, 0, 10)
Vertex_49a = geompy.MakeVertex(0.65, 30, -10)
Box_2a = geompy.MakeBoxTwoPnt(Vertex_41a, Vertex_49a)
Fuse_3a = geompy.MakeFuseList([Thickness_3, Extrusion_7a], True, True)
Fuse_4a = geompy.MakeFuseList([Thickness_1, Mirror_1a], True, True)
Cut_2a_Thickness_2 = geompy.MakeCutList(Cut_1, [Box_2a], True)
Cut_2a_Thickness_3 = geompy.MakeCutList(Fuse_3a, [Box_2a], True)
Cut_2a_Thickness_1 = geompy.MakeCutList(Fuse_4a, [Box_2a], True)
Vertex_41 = geompy.MakeVertex(6.308, 0, 0)
Vertex_42 = geompy.MakeVertex(18.15, -1, 0)
Vertex_45 = geompy.MakeVertex(18.15, -59.863, 0)
Vertex_46 = geompy.MakeVertex(6.308, -59.863, 0)
Vertex_49 = geompy.MakeVertex(7.308, -1, 0)
Vertex_50 = geompy.MakeVertex(6.308, -1, 0)

print('made fins')

Line_9 = geompy.MakeLineTwoPnt(Vertex_50, Vertex_49)
Line_10 = geompy.MakeLineTwoPnt(Vertex_49, Vertex_42)
Line_39 = geompy.MakeLineTwoPnt(Vertex_42, Vertex_45)
Line_40 = geompy.MakeLineTwoPnt(Vertex_45, Vertex_46)
Line_41 = geompy.MakeLineTwoPnt(Vertex_46, Vertex_50)
Wire_5 = geompy.MakeWire([Line_9, Line_10, Line_39, Line_40, Line_41], 1e-07)
Face_13 = geompy.MakeFaceWires([Wire_5], 1)
Extrusion_7 = geompy.MakePrismVecH2Ways(Face_13, OZ_1, 0.125)
Vertex_57 = geompy.MakeVertex(17.979, -59.515, 0)
Vertex_58 = geompy.MakeVertex(19.979, -57.515, 0)
Vertex_59 = geompy.MakeVertex(15.979, -61.515, 0)
Vertex_60 = geompy.MakeVertex(19.979, -61.515, 0)
Line_42 = geompy.MakeLineTwoPnt(Vertex_60, Vertex_58)
Line_43 = geompy.MakeLineTwoPnt(Vertex_57, Vertex_58)
Line_44 = geompy.MakeLineTwoPnt(Vertex_57, Vertex_59)
Line_45 = geompy.MakeLineTwoPnt(Vertex_59, Vertex_60)
Face_15 = geompy.MakeFaceWires([Line_42, Line_43, Line_44, Line_45], 1)
Revolution_4 = geompy.MakeRevolution2Ways(Face_15, OY_1, 10*math.pi/180.0)
Cut_XX = geompy.MakeCutList(Extrusion_7, [Revolution_4], True)
Rotation_5 = geompy.MakeRotation(Cut_XX, OY_1, 45*math.pi/180.0)
Rotation_6 = geompy.MakeRotation(Rotation_5, OY_1, 90*math.pi/180.0)
Rotation_7 = geompy.MakeRotation(Rotation_6, OY_1, 90*math.pi/180.0)
Rotation_8 = geompy.MakeRotation(Rotation_7, OY_1, 90*math.pi/180.0)
Fuse_2 = geompy.MakeFuseList([Cut_3, Cut_2a_Thickness_1, Cut_2a_Thickness_3, Thickness_4, Cut_2a_Thickness_2, Rotation_5, Rotation_6, Rotation_7, Rotation_8], False, False)
geomObj_252 = geompy.MakeVertex(0, 1, 0)
geomObj_253 = geompy.MakeCircle(geomObj_252, OY_1, 25)
geomObj_254 = geompy.MakeFaceWires([geomObj_253], 1)
geomObj_255 = geompy.MakePrismVecH(geomObj_254, OY_1, -70)
Cut_extra = geompy.MakeCutList(Fuse_2, [geomObj_255], True)
geomObj_256 = geompy.MakeVertex(-5, 25, 0)
geomObj_257 = geompy.MakeCircle(geomObj_256, OY_1, 24)
geomObj_258 = geompy.MakeFaceWires([geomObj_257], 1)
geomObj_259 = geompy.MakePrismVecH(geomObj_258, OY_1, -109.314)
Vertex_51 = geompy.MakeVertex(250, 23.46, -50)
Vertex_52 = geompy.MakeVertex(-250, 50, 50)
Box_2 = geompy.MakeBoxTwoPnt(Vertex_51, Vertex_52)
geomObj_260 = geompy.MakeCutList(geomObj_259, [Cut_extra], True)
geomObj_261 = geompy.MakeCutList(geomObj_260, [Box_2], True)
geompy.addToStudy(geomObj_261, 'geomObj_261')
[geomObj_262,geomObj_263,geomObj_264,geomObj_265,geomObj_266,geomObj_267,geomObj_268,geomObj_269,geomObj_270,geomObj_271,geomObj_272,geomObj_273,geomObj_274,geomObj_275,geomObj_276,geomObj_277,geomObj_278,geomObj_279,geomObj_280,geomObj_281, geomObj_282, geomObj_283, geomObj_284, geomObj_285, geomObj_286, geomObj_287, geomObj_288, geomObj_289, geomObj_290, geomObj_291, geomObj_292] = geompy.ExtractShapes(geomObj_261, geompy.ShapeType["FACE"], True)
Fuse_4 = geompy.MakeFuseList([Extrusion_3, Extrusion_4], True, True)

print('fuse 4')

Fuse_5 = geompy.MakeFuseList([geomObj_259, Fuse_4], True, True)
Cut_4 = geompy.MakeCutList(Fuse_5, [Fuse_2], True)
Cut_2 = geompy.MakeCutList(Cut_4, [Box_2], True)
Vertex_53 = geompy.MakeVertex(100, 23.46, -12.037532)
Vertex_54 = geompy.MakeVertex(100, 23.46, 12.037532)
Vertex_55 = geompy.MakeVertex(15.762895, 23.46, -12.037532)
Vertex_56 = geompy.MakeVertex(15.762895, 23.46, 12.037532)
Line_1 = geompy.MakeLineTwoPnt(Vertex_53, Vertex_54)
Line_2 = geompy.MakeLineTwoPnt(Vertex_54, Vertex_56)
Line_3 = geompy.MakeLineTwoPnt(Vertex_53, Vertex_55)
Line_4 = geompy.MakeLineTwoPnt(Vertex_55, Vertex_56)
Face_14 = geompy.MakeFaceWires([Line_1, Line_2, Line_3, Line_4], 1)
Cut_5 = geompy.MakeCutList(Face_14, [Extrusion_1], True)
Fuse_7 = geompy.MakeFuseList([Cut_5, geomObj_287], True, True)
Fuse_3 = geompy.MakePrismVecH(Fuse_7, OY_1, -8.932818961545085)
Cut_6 = geompy.MakeCutList(Cut_2, [Fuse_3], True)

print('cut 6')

Extrusion_9 = geompy.MakePrismVecH(Fuse_7, OY_1, -21.5)
Extrusion_10 = geompy.MakePrismVecH(Fuse_7, OY_1, -22)
Cut_7 = geompy.MakeCutList(Cut_2, [Extrusion_10], True)

print('cut 7')

trapGroup = geompy.GetInPlace(Cut_7, Fuse_upper, True)
trapGroupSubShapes = geompy.SubShapeAll(trapGroup, geompy.ShapeType["FACE"])

trap_surface = geompy.CreateGroup(Cut_7, geompy.ShapeType["FACE"])

trapGroupIDs = [geompy.GetSubShapeID(Cut_7, face) for face in trapGroupSubShapes]

geompy.UnionIDs(trap_surface, trapGroupIDs)

print('made trap group')

#WALLS, INLET, OUTLET ARE SAME EVERY TIME
#load in step files from previous sims. 

inlet_step_1 = geompy.ImportSTEP("/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/groupCreationStepFiles/inlet.step", False, True)
outlet_step_1 = geompy.ImportSTEP("/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/groupCreationStepFiles/outlet.step", False, True)
free_surface_step_1 = geompy.ImportSTEP("/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/groupCreationStepFiles/free_surface.step", False, True)

inletFace = geompy.GetInPlace(Cut_7, inlet_step_1, True)

inletSubShape = geompy.SubShapeAll(inletFace, geompy.ShapeType["FACE"])
inlet = geompy.CreateGroup(Cut_7, geompy.ShapeType["FACE"])

inletID = [geompy.GetSubShapeID(Cut_7, face) for face in inletSubShape]

geompy.UnionIDs(inlet, inletID)

#fullHood

# hoodFaces = geompy.GetInPlace(Cut_7, fullHood, True)
# hoodFacesSubShapes = geompy.SubShapeAll(hoodFaces, geompy.ShapeType["FACE"])
# hoodGroup = geompy.CreateGroup(Cut_7, geompy.ShapeType["FACE"])
# hoodGroupIDs = [geompy.GetSubShapeID(Cut_7, face) for face in hoodFacesSubShapes]
# geompy.UnionIDs(hoodGroup, hoodGroupIDs)



outletFace = geompy.GetInPlace(Cut_7, outlet_step_1, True)

outletSubShape = geompy.SubShapeAll(outletFace, geompy.ShapeType["FACE"])

outletID = [geompy.GetSubShapeID(Cut_7, face) for face in outletSubShape]

outlet = geompy.CreateGroup(Cut_7, geompy.ShapeType["FACE"])
geompy.UnionIDs(outlet, outletID)

freeSurfaceFace = geompy.GetInPlace(Cut_7, free_surface_step_1, True)
freeSurfaceSubShape = geompy.SubShapeAll(freeSurfaceFace, geompy.ShapeType["FACE"])
freeSurfaceID = [geompy.GetSubShapeID(Cut_7, face) for face in freeSurfaceSubShape]
free_surface = geompy.CreateGroup(Cut_7, geompy.ShapeType["FACE"])
geompy.UnionIDs(free_surface, freeSurfaceID)

#making wall group:

allFaces = geompy.ExtractShapes(Cut_7, geompy.ShapeType["FACE"], True)

walls = geompy.CreateGroup(Cut_7, geompy.ShapeType["FACE"])

geompy.UnionList(walls, allFaces)

# allFacesID = [geompy.GetSubShapeID(Cut_7, face) for face in freeSurfaceSubShape]
# trapFaces = geompy.ExtractShapes(trap_surface, geompy.ShapeType["FACE"], True)
# inletFaces = geompy.ExtractShapes(inlet, geompy.ShapeType["FACE"], True)
# outletFaces = geompy.ExtractShapes(outlet, geompy.ShapeType["FACE"], True)
# freeSurfaceFaces = geompy.ExtractShapes(free_surface, geompy.ShapeType["FACE"], True)
# hoodFaces2 = geompy.ExtractShapes(hoodGroup, geompy.ShapeType["FACE"], True)


# print('trap faces', trapFaces)

# print('free surface', freeSurfaceFaces)

# print('inlet', inletFaces)

# usedFaces = trapFaces + inletFaces + outletFaces + freeSurfaceFaces 

# print('used faces', usedFaces)

# usedFacesCompound = geompy.MakeCompound(usedFaces)

# remainingFaces = geompy.MakeCutList(Cut_7, usedFacesCompound)

# print('remaining faces', remainingFaces)

# # remainingFaces = [face for face in allFaces if face not in usedFaces]

# remainingFacesID = [geompy.GetSubShapeID(Cut_7, face) for face in remainingFaces]

# walls = geompy.CreateGroup(Cut_7, geompy.ShapeType["FACE"])
# geompy.UnionIDs(walls, remainingFacesID)


# [trap_surface, outlet, inlet, walls] = geompy.GetExistingSubObjects(Cut_7, False)
print('made wall group')
Revolution_1 = Revolution_1
Extrusion_8 = Fuse_3
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( O_1, 'O' )
geompy.addToStudy( OX_1, 'OX' )
geompy.addToStudy( OY_1, 'OY' )
geompy.addToStudy( OZ_1, 'OZ' )
geompy.addToStudy( Vertex_1, 'Vertex_1' )
geompy.addToStudy( Circle_1, 'Circle_1' )
geompy.addToStudy( Face_1, 'Face_1' )
geompy.addToStudy( Extrusion_1, 'Extrusion_1' )
geompy.addToStudy( Extrusion_2, 'Extrusion_2' )
geompy.addToStudy( Vertex_0, 'Vertex_0' )
geompy.addToStudy( Circle_2, 'Circle_2' )
geompy.addToStudy( Face_2, 'Face_2' )
geompy.addToStudy( Extrusion_3, 'Extrusion_3' )
geompy.addToStudy( Extrusion_4, 'Extrusion_4' )
geompy.addToStudy( Vertex_2, 'Vertex_2' )
geompy.addToStudy( Vertex_3, 'Vertex_3' )
geompy.addToStudy( Vertex_4, 'Vertex_4' )
geompy.addToStudy( Vertex_5, 'Vertex_5' )
geompy.addToStudy( Vertex_6, 'Vertex_6' )
geompy.addToStudy( Vertex_7, 'Vertex_7' )
geompy.addToStudy( Vertex_8, 'Vertex_8' )
geompy.addToStudy( Vertex_9, 'Vertex_9' )
geompy.addToStudy( Vertex_10, 'Vertex_10' )
geompy.addToStudy( Vertex_11, 'Vertex_11' )

geompy.addToStudy( upper_wire, 'upper_wire' )
geompy.addToStudy( Fuse_upper, 'Fuse_upper' )
geompy.addToStudy( Vertex_36, 'Vertex_36' )
geompy.addToStudy( Vertex_37, 'Vertex_37' )
geompy.addToStudy( Vertex_38, 'Vertex_38' )
geompy.addToStudy( Vertex_40, 'Vertex_40' )
geompy.addToStudy( Line_37, 'Line_37' )
geompy.addToStudy( Line_38, 'Line_38' )
geompy.addToStudy( Line_33, 'Line_33' )

geompy.addToStudy( line_1a, 'line_1a' )
geompy.addToStudy( Translation_2, 'Translation_2' )
geompy.addToStudy( Translation_3, 'Translation_3' )
geompy.addToStudy( Translation_4, 'Translation_4' )
geompy.addToStudy( Translation_5, 'Translation_5' )
geompy.addToStudy( Translation_6, 'Translation_6' )
geompy.addToStudy( Translation_7, 'Translation_7' )
geompy.addToStudy( Translation_8, 'Translation_8' )
geompy.addToStudy( Translation_9, 'Translation_9' )
geompy.addToStudy( Translation_10, 'Translation_10' )
geompy.addToStudy( Translation_11, 'Translation_11' )
geompy.addToStudy( Translation_12, 'Translation_12' )
geompy.addToStudy( Translation_13, 'Translation_13' )
geompy.addToStudy( Translation_14, 'Translation_14' )
geompy.addToStudy( Translation_15, 'Translation_15' )
geompy.addToStudy( Line_16a, 'Line_16a' )
geompy.addToStudy( Line_17a, 'Line_17a' )
geompy.addToStudy( Line_18a, 'Line_18a' )
geompy.addToStudy( Line_19a, 'Line_19a' )
geompy.addToStudy( Line_20a, 'Line_20a' )
geompy.addToStudy( Line_21a, 'Line_21a' )
geompy.addToStudy( Line_22a, 'Line_22a' )
geompy.addToStudy( Line_23a, 'Line_23a' )
geompy.addToStudy( Line_24a, 'Line_24a' )
geompy.addToStudy( Line_25a, 'Line_25a' )
geompy.addToStudy( wire_big, 'wire_big' )

geompy.addToStudy( Revolution_1, 'Revolution_1' )
geompy.addToStudy( Vertex_43, 'Vertex_43' )
geompy.addToStudy( Vertex_44, 'Vertex_44' )
geompy.addToStudy( Vertex_47, 'Vertex_47' )
geompy.addToStudy( Vertex_48, 'Vertex_48' )
geompy.addToStudy( Line_5, 'Line_5' )
geompy.addToStudy( Line_6, 'Line_6' )
geompy.addToStudy( Line_7, 'Line_7' )
geompy.addToStudy( Line_8, 'Line_8' )
geompy.addToStudy( Wire_6, 'Wire_6' )
geompy.addToStudy( Face_12, 'Face_12' )
geompy.addToStudy( Extrusion_6, 'Extrusion_6' )
geompy.addToStudy( Rotation_1, 'Rotation_1' )
geompy.addToStudy( Fuse_6, 'Fuse_6' )
geompy.addToStudy( Translation_16, 'Translation_16' )
geompy.addToStudy( Translation_17, 'Translation_17' )
geompy.addToStudy( Translation_18, 'Translation_18' )
geompy.addToStudy( Translation_19, 'Translation_19' )
geompy.addToStudy( Translation_20, 'Translation_20' )
geompy.addToStudy( Translation_21, 'Translation_21' )
geompy.addToStudy( Translation_22, 'Translation_22' )
geompy.addToStudy( Translation_23, 'Translation_23' )
geompy.addToStudy( Translation_24, 'Translation_24' )
geompy.addToStudy( Translation_25, 'Translation_25' )
geompy.addToStudy( Translation_26, 'Translation_26' )
geompy.addToStudy( Translation_27, 'Translation_27' )
geompy.addToStudy( Translation_28, 'Translation_28' )
geompy.addToStudy( Translation_29, 'Translation_29' )
geompy.addToStudy( Vertex_31, 'Vertex_31' )
geompy.addToStudy( Vertex_32, 'Vertex_32' )
geompy.addToStudy( Box_1, 'Box_1' )
geompy.addToStudy( Cut_3, 'Cut_3' )
geompy.addToStudy( Vertex_14, 'Vertex_14' )
geompy.addToStudy( Vertex_15, 'Vertex_15' )
geompy.addToStudy( Vertex_24, 'Vertex_24' )
geompy.addToStudy( Vertex_25, 'Vertex_25' )
geompy.addToStudy( Vertex_26, 'Vertex_26' )
geompy.addToStudy( Vertex_27, 'Vertex_27' )
geompy.addToStudy( Vertex_28, 'Vertex_28' )
geompy.addToStudy( Vertex_29, 'Vertex_29' )
geompy.addToStudy( Line_19, 'Line_19' )
geompy.addToStudy( Line_20, 'Line_20' )
geompy.addToStudy( Line_21, 'Line_21' )
geompy.addToStudy( Line_22, 'Line_22' )
geompy.addToStudy( Line_23, 'Line_23' )
geompy.addToStudy( Line_24, 'Line_24' )
geompy.addToStudy( Line_25, 'Line_25' )
geompy.addToStudy( Line_26, 'Line_26' )
geompy.addToStudy( Line_27, 'Line_27' )
geompy.addToStudy( Line_28, 'Line_28' )
geompy.addToStudy( Line_29, 'Line_29' )
geompy.addToStudy( Face_6, 'Face_6' )
geompy.addToStudy( Face_7, 'Face_7' )
geompy.addToStudy( Face_8, 'Face_8' )
geompy.addToStudy( Face_9, 'Face_9' )
geompy.addToStudy( Fuse_1, 'Fuse_1' )
geompy.addToStudy( Thickness_1, 'Thickness_1' )
geompy.addToStudy( Thickness_2, 'Thickness_2' )
geompy.addToStudy( Thickness_3, 'Thickness_3' )
geompy.addToStudy( Vertex_30, 'Vertex_30' )
geompy.addToStudy( Vertex_33, 'Vertex_33' )
geompy.addToStudy( Line_30, 'Line_30' )
geompy.addToStudy( Line_31, 'Line_31' )
geompy.addToStudy( Line_32, 'Line_32' )
geompy.addToStudy( Face_10, 'Face_10' )
geompy.addToStudy( Thickness_4, 'Thickness_4' )
geompy.addToStudy( Circle_3, 'Circle_3' )
geompy.addToStudy( Face_11, 'Face_11' )
geompy.addToStudy( Extrusion_5, 'Extrusion_5' )
geompy.addToStudy( Cut_1, 'Cut_1' )
geompy.addToStudy( Vertex_42a, 'Vertex_42a' )
geompy.addToStudy( Vertex_45a, 'Vertex_45a' )
geompy.addToStudy( Vertex_46a, 'Vertex_46a' )
geompy.addToStudy( Line_39a, 'Line_39a' )
geompy.addToStudy( Line_40a, 'Line_40a' )
geompy.addToStudy( Line_41a, 'Line_41a' )
geompy.addToStudy( Face_13a, 'Face_13a' )
geompy.addToStudy( Extrusion_7a, 'Extrusion_7a' )
geompy.addToStudy( Plane_1a, 'Plane_1a' )
geompy.addToStudy( Mirror_1a, 'Mirror_1a' )
geompy.addToStudy( Vertex_41a, 'Vertex_41a' )
geompy.addToStudy( Vertex_49a, 'Vertex_49a' )
geompy.addToStudy( Box_2a, 'Box_2a' )
geompy.addToStudy( Fuse_3a, 'Fuse_3a' )
geompy.addToStudy( Fuse_4a, 'Fuse_4a' )
geompy.addToStudy( Cut_2a_Thickness_2, 'Cut_2a_Thickness_2' )
geompy.addToStudy( Cut_2a_Thickness_3, 'Cut_2a_Thickness_3' )
geompy.addToStudy( Cut_2a_Thickness_1, 'Cut_2a_Thickness_1' )
geompy.addToStudy( Vertex_41, 'Vertex_41' )
geompy.addToStudy( Vertex_42, 'Vertex_42' )
geompy.addToStudy( Vertex_45, 'Vertex_45' )
geompy.addToStudy( Vertex_46, 'Vertex_46' )
geompy.addToStudy( Vertex_49, 'Vertex_49' )
geompy.addToStudy( Vertex_50, 'Vertex_50' )
geompy.addToStudy( Line_9, 'Line_9' )
geompy.addToStudy( Line_10, 'Line_10' )
geompy.addToStudy( Line_39, 'Line_39' )
geompy.addToStudy( Line_40, 'Line_40' )
geompy.addToStudy( Line_41, 'Line_41' )
geompy.addToStudy( Wire_5, 'Wire_5' )
geompy.addToStudy( Face_13, 'Face_13' )
geompy.addToStudy( Extrusion_7, 'Extrusion_7' )
geompy.addToStudy( Vertex_57, 'Vertex_57' )
geompy.addToStudy( Vertex_58, 'Vertex_58' )
geompy.addToStudy( Vertex_59, 'Vertex_59' )
geompy.addToStudy( Vertex_60, 'Vertex_60' )
geompy.addToStudy( Line_42, 'Line_42' )
geompy.addToStudy( Line_43, 'Line_43' )
geompy.addToStudy( Line_44, 'Line_44' )
geompy.addToStudy( Line_45, 'Line_45' )
geompy.addToStudy( Face_15, 'Face_15' )
geompy.addToStudy( Revolution_4, 'Revolution_4' )
geompy.addToStudy( Cut_XX, 'Cut_XX' )
geompy.addToStudy( Rotation_5, 'Rotation_5' )
geompy.addToStudy( Rotation_6, 'Rotation_6' )
geompy.addToStudy( Rotation_7, 'Rotation_7' )
geompy.addToStudy( Rotation_8, 'Rotation_8' )
geompy.addToStudy( Fuse_2, 'Fuse_2' )
geompy.addToStudy( Cut_extra, 'Cut_extra' )
geompy.addToStudy( Vertex_51, 'Vertex_51' )
geompy.addToStudy( Vertex_52, 'Vertex_52' )
geompy.addToStudy( Box_2, 'Box_2' )
geompy.addToStudy( Fuse_4, 'Fuse_4' )
geompy.addToStudy( Fuse_5, 'Fuse_5' )
geompy.addToStudy( Cut_4, 'Cut_4' )
geompy.addToStudy( Cut_2, 'Cut_2' )
geompy.addToStudy( Vertex_53, 'Vertex_53' )
geompy.addToStudy( Vertex_54, 'Vertex_54' )
geompy.addToStudy( Vertex_55, 'Vertex_55' )
geompy.addToStudy( Vertex_56, 'Vertex_56' )
geompy.addToStudy( Line_1, 'Line_1' )
geompy.addToStudy( Line_2, 'Line_2' )
geompy.addToStudy( Line_3, 'Line_3' )
geompy.addToStudy( Line_4, 'Line_4' )
geompy.addToStudy( Face_14, 'Face_14' )
geompy.addToStudy( Cut_5, 'Cut_5' )
geompy.addToStudy( Fuse_7, 'Fuse_7' )
geompy.addToStudy( Fuse_3, 'Fuse_3' )
geompy.addToStudy( Cut_6, 'Cut_6' )
geompy.addToStudy( Extrusion_9, 'Extrusion_9' )
geompy.addToStudy( Extrusion_10, 'Extrusion_10' )
geompy.addToStudy( Cut_7, 'Cut_7' )
geompy.addToStudyInFather( Cut_7, trap_surface, 'trap_surface' )
geompy.addToStudyInFather( Cut_7, outlet, 'outlet' )
geompy.addToStudyInFather( Cut_7, inlet, 'inlet' )
geompy.addToStudyInFather( Cut_7, walls, 'walls' )
# geompy.addToStudyInFather(Cut_7, hoodGroup, 'hoodGroup')
geompy.addToStudyInFather(Cut_7, free_surface, 'free_surface')

###
### SMESH component
###
print('meshing')
import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New()
#smesh.SetEnablePublish( False ) # Set to False to avoid publish in study if not needed or in some particular situations:
                                 # multiples meshes built in parallel, complex and numerous mesh edition (performance)

#hyp_19.SetLength( 0.2 ) ### not created Object
NETGEN_2D_Parameters_1 = smesh.CreateHypothesisByAverageLength( 'NETGEN_Parameters_2D', 'NETGENEngine', 0.2, 0 )
NETGEN_1D_2D = smesh.CreateHypothesis('NETGEN_2D', 'NETGENEngine')
#hyp_3.SetLength( 7.59276 ) ### not created Object
NETGEN_2D_Parameters_2 = smesh.CreateHypothesisByAverageLength( 'NETGEN_Parameters_2D', 'NETGENEngine', 7.59276, 0 )
NETGEN_2D_Parameters_3 = smesh.CreateHypothesisByAverageLength( 'NETGEN_Parameters_2D', 'NETGENEngine', 16.0292, 0 )
NETGEN_2D_Parameters_4 = smesh.CreateHypothesis('NETGEN_Parameters_2D', 'NETGENEngine')
NETGEN_2D_Parameters_4.SetMaxSize( 0.2 )
NETGEN_2D_Parameters_4.SetMinSize( 0.15 )
NETGEN_2D_Parameters_4.SetSecondOrder( 0 )
NETGEN_2D_Parameters_4.SetOptimize( 1 )
NETGEN_2D_Parameters_4.SetFineness( 2 )
NETGEN_2D_Parameters_4.SetChordalError( -1 )
NETGEN_2D_Parameters_4.SetChordalErrorEnabled( 0 )
NETGEN_2D_Parameters_4.SetUseSurfaceCurvature( 1 )
NETGEN_2D_Parameters_4.SetFuseEdges( 1 )
NETGEN_2D_Parameters_4.SetWorstElemMeasure( 0 )
NETGEN_2D_Parameters_4.SetQuadAllowed( 0 )
NETGEN_2D_Parameters_4.SetUseDelauney( 128 )
NETGEN_2D_Parameters_4.SetCheckChartBoundary( 3 )
wholeGeo = smesh.Mesh(Cut_7,'wholeGeo')
status = wholeGeo.AddHypothesis(NETGEN_2D_Parameters_4)
status = wholeGeo.AddHypothesis(NETGEN_1D_2D)
trap_surface_1 = wholeGeo.GroupOnGeom(trap_surface,'trap_surface',SMESH.FACE)
outlet_1 = wholeGeo.GroupOnGeom(outlet,'outlet',SMESH.FACE)
inlet_1 = wholeGeo.GroupOnGeom(inlet,'inlet',SMESH.FACE)
walls_1 = wholeGeo.GroupOnGeom(walls,'walls',SMESH.FACE)
# hoodGroup_1 = wholeGeo.GroupOnGeom(hoodGroup, 'hoodGroup', SMESH.FACE)
free_surface_1 = wholeGeo.GroupOnGeom(free_surface, 'free_surface', SMESH.FACE)
isDone = wholeGeo.Compute()

#hoodGroup
[ trap_surface_1, outlet_1, inlet_1, walls_1, free_surface_1 ] = wholeGeo.GetGroups()
try:
  wholeGeo.ExportSTL( r'/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/stlGroupsNew/trap_surface.stl', 1, trap_surface_1)
  pass
except:
  print('ExportPartToSTL() failed. Invalid file name?')
try:
  wholeGeo.ExportSTL( r'/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/stlGroupsNew/outlet.stl', 1, outlet_1)
  pass
except:
  print('ExportPartToSTL() failed. Invalid file name?')
try:
  wholeGeo.ExportSTL( r'/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/stlGroupsNew/inlet.stl', 1, inlet_1)
  pass
except:
  print('ExportPartToSTL() failed. Invalid file name?')
# try:
#   wholeGeo.ExportSTL( r'/home/bm424/Projects/Catmull-Clark/stlGroups/walls.stl', 1, walls_1)
#   pass
# except:
#   print('ExportPartToSTL() failed. Invalid file name?')
# try:
#   wholeGeo.ExportSTL( r'/home/bm424/Projects/Catmull-Clark/stlGroups/hoodGroup.stl', 1, hoodGroup_1)
#   pass
# except:
#   print('ExportPartToSTL() failed. Invalid file name?')
try:
  wholeGeo.ExportSTL( r'/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/stlGroupsNew/free_surface.stl', 1, free_surface_1)
  pass
except:
  print('ExportPartToSTL() failed. Invalid file name?')

allCut = wholeGeo.GetMesh().CutListOfGroups( [ walls_1 ], [ trap_surface_1, outlet_1, inlet_1, free_surface_1 ], 'allCut' )
try:
  wholeGeo.ExportSTL( r'/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/stlGroupsNew/walls.stl', 1, allCut)
  pass
except:
  print('ExportPartToSTL() failed. Invalid file name?')

fullHood_1 = smesh.Mesh(fullHood,'fullHood')
NETGEN_1D_2D_1 = fullHood_1.Triangle(algo=smeshBuilder.NETGEN_1D2D)
isDone = fullHood_1.Compute()
try:
  fullHood_1.ExportSTL( r'/home/bm424/OpenFOAM/bm424-v2312/run/pinecone/salomeToOF/stlGroupsNew/fullHood.stl', 1 )
  pass
except:
  print('ExportSTL() failed. Invalid file name?')

## Set names of Mesh objects
smesh.SetName(NETGEN_1D_2D, 'NETGEN 1D-2D')
smesh.SetName(NETGEN_2D_Parameters_2, 'NETGEN 2D Parameters_2')
smesh.SetName(NETGEN_2D_Parameters_3, 'NETGEN 2D Parameters_3')
smesh.SetName(NETGEN_2D_Parameters_1, 'NETGEN 2D Parameters_1')
smesh.SetName(NETGEN_2D_Parameters_4, 'NETGEN 2D Parameters_4')
smesh.SetName(wholeGeo.GetMesh(), 'wholeGeo')
smesh.SetName(outlet_1, 'outlet')
smesh.SetName(inlet_1, 'inlet')
smesh.SetName(trap_surface_1, 'trap_surface')
smesh.SetName(walls_1, 'walls')

endTime = time.time()

print('elapsed time =', (endTime-startTime))

if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
