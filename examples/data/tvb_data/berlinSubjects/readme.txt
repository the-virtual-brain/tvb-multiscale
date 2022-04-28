
## TVB Import Files Readme
This Readme describes the content of the different output files
of the Berlin empirical data processing pipeline, meant to be imported into TVB.

We provide two subjects scans: DH and QL, and for each of them 7 separate files to be imported in TVB

## 1. [subID]_[date]_Connectivity.zip
A single zip-File that contains seven single ASCII-files describing the connectivity between different predefined
brain areas. The files are.
-- 1.) area.txt - The areal dimensions of each different region
-- 2.) centres.txt - Labels and XYZ-coordinates for the most central vertices of the different regions
-- 3.) cortical.txt - A binary list depicting wether the corresponding area is a cortical (1) or a subcortical (0) one
-- 4.) hemisphere.txt - An optional binary list depicting the hemisphere on which the corresponding region lies (0 = left; 1 = right)
-- 5.) orientation.txt - The vertex normals of the central vertices
-- 6.) tract.txt - The connectivity tract lengths in mm
-- 7.) weights.txt - The matrix describing the connectivity strength between the brain areas. 
The unit is somehow arbitrary in a sense that it only reflects the number of distinct connections to be found between the different 
voxels of the brain regions when applying a probabilistic fibertracking algorithm, divided by the total amount of tracks 
incoming to the separate voxels to reflect the fact that each part of the Greymatter/Whitematter-interface is only
capable of holding a certain amount of fibers, distributing his resources amongst them.

## 2. [subID]_[date]_Surface_Cortex.zip
The triangulated surface mesh of the individual cortical surface. The zip-file contains three different ASCII-files:
-- 1.) normals.txt - The vertex-normals of the mesh vertices.
-- 2.) triangles.txt - Three columns, describing which vertices form the triangles of the mesh. Hence each entry hold the index to a vertex.
-- 3.) vertices.txt - The XYZ-coordinates of each vertex.

## 3. [subID]_[date]_Surface_EEGCAP.zip
The triangulated surface mesh of the individual skin surface. This mesh has to be used as the "EEG cap" inside TVB to project the electrodes on.
The zip-file contains three different ASCII-files:
-- 1.) normals.txt - The vertex-normals of the mesh vertices.
-- 2.) triangles.txt - Three columns, describing which vertices form the triangles of the mesh. Hence each entry hold the index to a vertex.
-- 3.) vertices.txt - The XYZ-coordinates of each vertex.

## 4. [subID]_[date]_Surface_Face.zip
The triangulated surface mesh of the individual skin surface. The zip-file contains three different ASCII-files:
-- 1.) normals.txt - The vertex-normals of the mesh vertices.
-- 2.) triangles.txt - Three columns, describing which vertices form the triangles of the mesh. Hence each entry hold the index to a vertex.
-- 3.) vertices.txt - The XYZ-coordinates of each vertex.

## 5. [subID]_[date]_EEGLocations.txt
A single ASCII-file, separated into four main columns: 
The first column consists of labels for the EEG-sensors
The next three columns are the XYZ-coordinates corresponding to the headsurface-mesh.
The vectors which they form where normalized onto a length of 1.

## 6. [subID]_[date]_ProjectionMatrix.mat
A MATLAB-storage file including the matrix which describes the forward-solution for the cortical surface mesh i.e.
how the electrical signal of (artifical) neuronal dipoles at the positions of the surface vertices with a orientation perpendicular
to these points (i.e.the vertex-normals) propagates through a multi-layer headmodel onto the EEG sensors.
Hence the dimensions of this matrix are #EEGSensors X #CortexMeshVertices
Before importing this ProjectionMatrix in TVB, you need to have the Connectivity, as well and the EEG Sensors for this particular subject imported (data at number 1 and 5). Also when importing the ProjectionMatrix, in the user interface of TVB, you need to make sure the correct Connectivity entity and EEG Sensors for this subjects are selected.

## 7. [subID]_[date]_RegionMapping.txt
A single ASCII-file consisting of #CortexMeshVertices entries, depicting which region each vertex of the cortical mesh belongs to.
Before importing this RegionMapping data in TVB, you need to import the Connectivity and the CorticalSurface for the subject. You also need to match them correctly in the user interface when importing the RegionMapping.




For Subject QL, also an empirical BOLD signal can be found: QL_BOLD_regiontimecourse.mat
To import this file, try "TimeSeries MAT" importer of TVB, and specify "QL_BOLD_regiontimecourse" for field "Matlab DataSet Name".
