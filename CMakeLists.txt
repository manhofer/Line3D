cmake_minimum_required(VERSION 2.6)

set (CMAKE_BUILD_TYPE Release)
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

##-----------------------------------------------------------------------------
## OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
set (EXTRA_INC_DIRS
    ${EXTRA_INC_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
set (EXTRA_LIBRARIES
    ${EXTRA_LIBRARIES}
    ${OpenCV_LIBS}
)

link_directories(${OPENCV_LIB_DIR})

##-----------------------------------------------------------------------------
## Eigen
FIND_PACKAGE(Eigen REQUIRED)

set (EXTRA_INC_DIRS
    ${EXTRA_INC_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}/unsupported
)
include_directories(${EIGEN3_INCLUDE_DIR})

##-----------------------------------------------------------------------------
## CUDA
  FIND_PACKAGE(CUDA REQUIRED)
  IF(CUDA_FOUND)
    set (EXTRA_INC_DIRS
        ${EXTRA_INC_DIRS}
        ${CUDA_INCLUDE_DIRS}
		${CUDA_SDK_INCLUDE_DIR}
    )
	cuda_include_directories(${EXTRA_INC_DIRS})
    set (EXTRA_LIBRARIES
        ${EXTRA_LIBRARIES}
        ${CUDA_LIBS}
        ${CUDA_cublas_LIBRARY}
    )
  ENDIF(CUDA_FOUND)

##-----------------------------------------------------------------------------
## Boost
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_STATIC_RUNTIME     OFF)
FIND_PACKAGE(Boost REQUIRED COMPONENTS serialization filesystem system)

if(Boost_FOUND)
        set (EXTRA_INC_DIRS
                ${EXTRA_INC_DIRS}
                ${Boost_INCLUDE_DIRS}
        )

          set (EXTRA_LIBRARIES
                ${EXTRA_LIBRARIES}
                ${Boost_LIBRARIES}
          )
endif(Boost_FOUND)

##-----------------------------------------------------------------------------
## LSD (external)
##------------------------------------------------------------------------------
SET(lsd_SOURCES lsd/lsd.cpp lsd/lsd_opencv.cpp lsd/lsd_wrap.cpp)
SET(lsd_HEADERS lsd/lsd.hpp lsd/lsd_opencv.hpp lsd/lsd_wrap.hpp)
add_library(line3D_lsd SHARED ${lsd_SOURCES} ${lsd_HEADERS})
target_link_libraries(line3D_lsd ${OpenCV_LIBS})

add_definitions(-frounding-math)

#---- combine external libs -----
set(ALL_LIBRARIES line3D_lsd ${EXTRA_LIBRARIES})

#---- Add Line3D library----
SET(Line3D_SOURCES line3D.cc view.cc sparsematrix.cc clustering.cc cudawrapper.cu)
SET(Line3D_HEADERS line3D.h view.h sparsematrix.h clustering.h universe.h segments.h serialization.h commons.h dataArray.h cudawrapper.h)

CUDA_ADD_LIBRARY(line3D SHARED ${Line3D_SOURCES} ${Line3D_HEADERS})
target_link_libraries(line3D ${ALL_LIBRARIES})

#----- Add main for bundler --------
add_executable(runLine3D_bundler main_bundler.cpp)
target_link_libraries(runLine3D_bundler line3D)
target_link_libraries(runLine3D_bundler ${ALL_LIBRARIES})

#----- Add main for VisualSfM --------
add_executable(runLine3D_vsfm main_vsfm.cpp)
target_link_libraries(runLine3D_vsfm line3D)
target_link_libraries(runLine3D_vsfm ${ALL_LIBRARIES})

