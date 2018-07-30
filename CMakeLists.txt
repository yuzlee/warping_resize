cmake_minimum_required(VERSION 3.3)

if (Linux)
    add_definitions("-std=c++11")
endif()

if (_DEBUG)
    add_definitions("-D_DEBUG")
    add_definitions("-g")
else ()
    if (Linux)
        add_definitions("-O2")
    endif ()
endif ()


if (WIN32)
    set(CMAKE_PREFIX_PATH D:/Resource/opencv/target;D:/Resource/eigen/target)
else ()
    set(CMAKE_PREFIX_PATH /opt/workspace/opencv/build)
endif ()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

macro(ocv_app EXEC_NAME)
    message("add exec: ${EXEC_NAME}")
    add_executable(${EXEC_NAME} ${ARGN})
    target_link_libraries(${EXEC_NAME} ${OpenCV_LIBS})
endmacro()

set(ALL_SOURCES
        3rdparty/lsd-1.5/lsd.cpp
        3rdparty/lsd-1.5/lsd.h

        src/lsd.hpp
        src/boundary_segment.hpp
        src/seam_carving.hpp

        src/interpolation.hpp
        src/line.hpp
        src/mesh.hpp
        src/optimizer.hpp
        src/global_warping.hpp

        src/timer.hpp
        src/util.hpp
        src/matrix_data.hpp
        )

ocv_app(resize ${ALL_SOURCES} src/warping_resize.cpp)

# ocv_app(test_resize ${ALL_SOURCES} test/test_resize.cpp)

# ocv_app(test_make_mask test/make_segment_mask.cpp)

# ocv_app(test_convert_mask test/convert_wasm_mask.cpp)

# ocv_app(test_roll_mat test/test_roll_shift.cpp)

# ocv_app(test_line 3rdparty/lsd-1.5/lsd.cpp test/test_line.cpp)

# ocv_app(test_mesh test/test_mesh.cpp)
