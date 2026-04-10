cmake_minimum_required(VERSION 3.16)
project(ThickTargetBrems)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(thick_target_brems thick_target_brems.cc)
target_link_libraries(thick_target_brems ${Geant4_LIBRARIES})
