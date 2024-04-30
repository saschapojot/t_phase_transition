cmake_minimum_required(VERSION 3.27)
project(t_phase_transition)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories("/home/polya/cppLocal/armadillo/include")



add_executable(run1d run1d.cpp 1d/1d.cpp)

set_source_files_properties(
        1d/1d.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)


set(Boost_NO_SYSTEM_PATHS YES)
if (Boost_NO_SYSTEM_PATHS)
    set(Boost_ROOT "/home/polya/cppLocal/boostBuild")
    set(Boost_INCLUDE_DIR "/home/polya/cppLocal/boostBuild/include")
    set(Boost_LIBRARY_DIR "/home/polya/cppLocal/boostBuild/lib")
endif (Boost_NO_SYSTEM_PATHS)

#find_package(Boost  REQUIRED COMPONENTS ALL)
find_package(Boost  REQUIRED serialization filesystem)
if(Boost_FOUND)
    message(STATUS "boost root is: ${Boost_ROOT}")
    message(STATUS "boost include path is : ${Boost_INCLUDE_DIR}")
    message(STATUS "boost library path is : ${Boost_LIBRARY_DIR}")
    message(STATUS "boost libraries is : ${Boost_LIBRARIES}")
    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${Boost_LIBRARY_DIR})
else()
    message(WARNING "boost not found.")
endif()


file(GLOB LIBS_SO "/home/polya/cppLocal/armadillo/lib/*.so")
target_link_libraries(run1d ${LIBS_SO})

target_link_libraries(run1d Boost::serialization)
target_link_libraries(run1d Boost::filesystem)