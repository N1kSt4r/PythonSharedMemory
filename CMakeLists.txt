cmake_minimum_required(VERSION 3.15)
project(PythonSharedMemory)

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_FLAGS " -pthread -O2 " )
set( CMAKE_POSITION_INDEPENDENT_CODE ON )
set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )


find_package( Python3 COMPONENTS Interpreter Development NumPy REQUIRED )
message( STATUS "Python: " ${Python3_EXECUTABLE} )
message( STATUS "        " ${Python3_INCLUDE_DIRS} )
message( STATUS "        " ${Python3_LIBRARIES} )
message( STATUS "        " ${Python3_NumPy_INCLUDE_DIRS} )

include_directories( ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )


set( Boost_USE_STATIC_LIBS   OFF )
set( Boost_USE_MULTITHREADED ON )

find_package( Boost COMPONENTS python3 numpy3 thread system REQUIRED )
message( STATUS "Boost:  " ${Boost_INCLUDE_DIR} )
message( STATUS "        " ${Boost_LIBRARY_DIRS} )
message( STATUS "        " ${Boost_LIBRARIES} )

include_directories( ${Boost_INCLUDE_DIR} )
add_definitions( ${Boost_DEFINITIONS} )


add_library( shared_store SHARED module.cpp )


target_link_libraries( shared_store ${Boost_LIBRARIES} ${Python3_LIBRARIES} librt.so )
target_include_directories( shared_store PRIVATE ${Python3_INCLUDE_DIRS} )
set_target_properties( shared_store PROPERTIES PREFIX "" )
set_target_properties( shared_store PROPERTIES SUFFIX ".so" )


add_executable( exe main.cpp )
target_link_libraries( exe shared_store )
