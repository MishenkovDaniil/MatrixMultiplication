cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++-12)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(matrix_project VERSION 0.1 LANGUAGES C)

option(BUILD_TESTS "Build unit tests" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(matrix STATIC
    src/matrix.c
    src/matrix_blocked_pthread.c
    src/matrix_pthreads.c
)

target_include_directories(matrix
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(matrix_demo
    src/main.c
)
add_executable(matrix_test
    tests/fill_test_array.c
)
add_executable(matrix_nthreads_test
    tests/fill_nthreads_arr.c
)

target_link_libraries(matrix_demo PRIVATE matrix)
target_link_libraries(matrix_test PRIVATE matrix)
target_link_libraries(matrix_nthreads_test PRIVATE matrix)
