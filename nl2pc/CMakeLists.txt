cmake_minimum_required(VERSION 3.14)
project(nl2pc LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "Found pybind11")
elseif(NOT pybind11_FOUND AND NOT TARGET pybind11::pybind11)
    message("pybind11 was not found: add pybind11 subdirectory")
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
        find_package(Git REQUIRED)
        message("initialize Git submodule: extern/pybind11")
        execute_process(COMMAND git submodule update --init extern/pybind11
                        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif()
    add_subdirectory(extern/pybind11)
endif()

find_package(SEAL QUIET)
if(SEAL_FOUND)
    message(STATUS "Found SEAL")
elseif(NOT SEAL_FOUND AND NOT TARGET SEAL::seal)
    message("SEAL was not found: add SEAL subdirectory")
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/SEAL/CMakeLists.txt")
        find_package(Git REQUIRED)
        message("initialize Git submodule: extern/SEAL")
        execute_process(COMMAND git submodule update --init extern/SEAL
                        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif()
    add_subdirectory(extern/SEAL)
endif()

add_subdirectory(src)