--------------------------------------------------------------------------------
Line3D - Line-based 3D Reconstruction Algorithm
--------------------------------------------------------------------------------
Written by: Manuel Hofer, hofer@icg.tugraz.at,
Institute for Computer Graphics and Vision,
Graz University of Technology,
AUSTRIA,
http://www.icg.tugraz.at/
--------------------------------------------------------------------------------
Corresponding Paper:
"Line3D: Efficient 3D Scene Abstraction for the Built Environment",
Manuel Hofer, Michael Maurer, Horst Bischof,
In Proceedings of the 37th German Conference on Pattern Recognition (GCPR),
2015.
--------------------------------------------------------------------------------

1, Requirements:
There is a CMakeLists.txt file included, which should enable you to easily 
build the Line3D library using CMake.

External dependencies:
- CUDA (>= 5.0)
- boost (>= 1.50.x)
- Eigen3
- OpenCV (>= 2.3.x)
- tclap

If all these libraries are properly installed, compiling should be no problem!
The version numbers above are only rough guesses with respect to what I am
using myself (it might as well work with much older/newer versions as well).

There are also two external components, which are integrated into the sourcecode
for your convenience:

- LSD - LineSegmentDetector (see copyright notice in lsd/lsd_opencv.hpp):
"A Fast Line Segment Detector with a False Detection Control",
R. von Gioi, J. Jakubowicz, J.M. Morel, G. Randall,
Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2010.

- Graph-based Segmentation (adopted from http://cs.brown.edu/~pff/segment/):
"Efficient Graph-based Image Segmentation",
P. Fezenszwalb, F. Huttenlocher,
International Journal of Computer Vision, 2004.

Note: I have only tried this application on Ubuntu (14.04)! Since all external
libraries should run on Windows as well it should be no problem to compile Line3D
there (adaptaions of the CMakeLists.txt might be necessary).

--------------------------------------------------------------------------------

2, Usage:
Line3D is a library which can be linked to any kind of C/C++ application.
However, for convenience there is an executable included, which can process
bundler (http://www.cs.cornell.edu/~snavely/bundler/) and VisualSfM
(http://ccwu.me/vsfm/) results.

To compute a line-based 3D model for a SfM reconstruction using bundler
or VisualSfM run the following executable:

./runLine3D_vsfm -i <path_to_folder_with_bundler.rd.out_file>

That's it! The result will be placed in the folder:
<path_to_folder_with_bundler.rd.out_file>/Line3D/

If you want to use Line3D within your own 3D reconstruction pipeline
you just need to either link it to your existing application, or create
a new main.cpp file (have a look at main_vsfm.cpp) which can process
your SfM output.

The following steps need to be followed to create 3D line models:

- create Line3D object: Line3D(...) [line3D.h]

- add images: void addImage(...) [line3D.h]
Call this function for each image in your SfM result. The method needs
the camera information (intrinsics, position) and a list of worldpoint IDs.
The 3D position of the worldpoints is irrelevant, this is only needed
to find visual neighbors among the images.

- compute 3D model: void compute3Dmodel(...) [line3D.h]
The algorithm now runs the matching and reconstruction steps. You can
specify if you want to perform a diffusion-based correspondence
optimization or not (default is off; usually not needed).

- get result: void getResult(...) [line3D.h]
The result contains the 3D lines and the corresponding 2D segment IDs
in the images (if needed, you can retrieve the coordinates using the
float4 getSegment2D(...) function).

If you have questions regarding this process, please have a look at
the main_vsfm.cpp or contact me.

--------------------------------------------------------------------------------

3, Parameters:
The executable (for VisualSfM/bundler) takes a number of command line parameters,
most of which are optional:

-i [string] - Input_Folder (required)
Path to the folder where the bundler.rd.out is located

-o [string] - Output_Folder
Output folder. If not specified, same as Input_Folder but with /Line3D/ attached
to it (will be created automatically).

-w [int] - Image_Max_Width
The maximum width (or height, if the image is in portrait format) to which the images
will be downscaled before line segment detection. By default this is set to 1920 (FullHD),
which should not be altered if you work with images >= 10Megapixels
(the detection takes less time and the results do not suffer).

-n [int] - Number_of_Visual_Neighbors
The number of images with which each image is matched. By default this is set to 12.
Since the matching is very fast I would not recommend to decrease this number.
Increasing it should only be done if you have a GPU with a large amount of memory!
If you set it to -1, all images which share worldpoints are used (not recommended!)

-a [float] - Reprojection_Error_Lower_Bound
-b [float] - Reprojection_Error_Upper_Bound
The error bounds (in pixels) for the affinity computation and clustering. These
values will be projected into 3D space for 3D similarity computation (please have
a look at the paper). If the results are too sparse, try increasing -b.

-g [float] - Sigma_Angle
-p [float] - Sigma_Position
Values for confidence estimation of 3D line hypotheses (again in pixels). Similar to
the two values above, but usually don't need to be changed (again, please have
a look at the paper).

-r [bool] - 3D_Verification
Verifies the confidence of 3D hypotheses in 3D space as well. Should always be switched
on (default), switching it off might lead to denser results but will also introduce
some outliers (most likely).

-d [bool] - Diffusion
Activates the diffusion process before clustering. Should not be used for extremely
large datasets (>1000 images) and can in general be left disabled.

-v [bool] - Verbose
Shows more debug output.

-l [bool] - Load_And_Store_Segments
If enabled, 2D line segments are stored on the harddrive and don't have to be
computed again when the testcase is re-run.

-e [bool] - Collinearity
If enabled (default), collinear but spatially distant segments are grouped
together and aligned.

-x [float] - Min_Baseline
To avoid that images with a very small baseline are matched, you can specify
the minimal required basedline here. This is the only value which is not scale
invariant! By default it is set to 0.25f, which should be sufficient for a
large number of scenarios.

--------------------------------------------------------------------------------

4, Results:
The default output format for the 3D line models is an .stl file. It is
placed in the output folder and named with respect to the choosen parameters.

Note: If you open this file in Meshlab (http://meshlab.sourceforge.net/)
you should see a window which states "Post-Open Processing". You have to
untick the option "Unify Duplicated Vertices" and klick OK. Then switch to
"Wireframe", "Hidden Lines" or "Flat Lines" modes and you should see a 
result.

In addition the result is also saved as a .txt file in the following format
(one line in the file per resulting 3D line):

n P1x P1y P1z Q1x Q1y Q1z ... Pnx Pny Pnz Qnx Qny Qnz m camID1 segID1 p1x p1y q1x q1y ... camIDm segIDm pmx pmy qmx qmy

The first "n" stands for the number of 3D segments on the 3D line.
It is followed by the coordinates of the start- (P) and endpoints (Q) of these 3D segments.
The "m" stands for the number of 2D residuals.
It is followed by the camera ID, segment ID and the 2D coordinates of the segments.

--------------------------------------------------------------------------------

If you have any questions or have found any bugs please contact me:
hofer@icg.tugraz.at

Best,
Manuel

