# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/wenhou/cmake-3.8.2-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/wenhou/cmake-3.8.2-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wenhou/cv_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wenhou/cv_ws/build

# Include any dependencies generated for this target.
include robot_vision/CMakeFiles/cv_kinect.dir/depend.make

# Include the progress variables for this target.
include robot_vision/CMakeFiles/cv_kinect.dir/progress.make

# Include the compile flags for this target's objects.
include robot_vision/CMakeFiles/cv_kinect.dir/flags.make

robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o: robot_vision/CMakeFiles/cv_kinect.dir/flags.make
robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o: /home/wenhou/cv_ws/src/robot_vision/src/cv_kinect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wenhou/cv_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o"
	cd /home/wenhou/cv_ws/build/robot_vision && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o -c /home/wenhou/cv_ws/src/robot_vision/src/cv_kinect.cpp

robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.i"
	cd /home/wenhou/cv_ws/build/robot_vision && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wenhou/cv_ws/src/robot_vision/src/cv_kinect.cpp > CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.i

robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.s"
	cd /home/wenhou/cv_ws/build/robot_vision && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wenhou/cv_ws/src/robot_vision/src/cv_kinect.cpp -o CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.s

robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o.requires:

.PHONY : robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o.requires

robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o.provides: robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o.requires
	$(MAKE) -f robot_vision/CMakeFiles/cv_kinect.dir/build.make robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o.provides.build
.PHONY : robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o.provides

robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o.provides.build: robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o


# Object files for target cv_kinect
cv_kinect_OBJECTS = \
"CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o"

# External object files for target cv_kinect
cv_kinect_EXTERNAL_OBJECTS =

/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: robot_vision/CMakeFiles/cv_kinect.dir/build.make
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libcv_bridge.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_core3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libimage_transport.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libmessage_filters.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libclass_loader.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/libPocoFoundation.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libdl.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libroslib.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/librospack.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libpython2.7.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libtinyxml.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libroscpp.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/librosconsole.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/librosconsole_log4cxx.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/librosconsole_backend_interface.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libxmlrpcpp.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libserial.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libroscpp_serialization.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/librostime.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libcpp_common.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_superres3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_face3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_img_hash3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_reg3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_tracking3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_shape3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_photo3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_viz3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_video3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_plot3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_text3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_dnn3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_flann3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_ml3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: /opt/ros/kinetic/lib/libopencv_core3.so.3.3.1
/home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect: robot_vision/CMakeFiles/cv_kinect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wenhou/cv_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect"
	cd /home/wenhou/cv_ws/build/robot_vision && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cv_kinect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
robot_vision/CMakeFiles/cv_kinect.dir/build: /home/wenhou/cv_ws/devel/lib/robot_vision/cv_kinect

.PHONY : robot_vision/CMakeFiles/cv_kinect.dir/build

robot_vision/CMakeFiles/cv_kinect.dir/requires: robot_vision/CMakeFiles/cv_kinect.dir/src/cv_kinect.cpp.o.requires

.PHONY : robot_vision/CMakeFiles/cv_kinect.dir/requires

robot_vision/CMakeFiles/cv_kinect.dir/clean:
	cd /home/wenhou/cv_ws/build/robot_vision && $(CMAKE_COMMAND) -P CMakeFiles/cv_kinect.dir/cmake_clean.cmake
.PHONY : robot_vision/CMakeFiles/cv_kinect.dir/clean

robot_vision/CMakeFiles/cv_kinect.dir/depend:
	cd /home/wenhou/cv_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wenhou/cv_ws/src /home/wenhou/cv_ws/src/robot_vision /home/wenhou/cv_ws/build /home/wenhou/cv_ws/build/robot_vision /home/wenhou/cv_ws/build/robot_vision/CMakeFiles/cv_kinect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : robot_vision/CMakeFiles/cv_kinect.dir/depend

