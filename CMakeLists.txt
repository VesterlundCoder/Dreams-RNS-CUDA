cmake_minimum_required(VERSION 3.18)
project(dreams_rns_cuda CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(DREAMS_ENABLE_GPU "Enable CUDA GPU support" ON)
option(DREAMS_ENABLE_TESTS "Enable tests" ON)

# Check for CUDA
if(DREAMS_ENABLE_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        add_definitions(-DDREAMS_HAS_GPU)
        message(STATUS "CUDA found - GPU support enabled")
    else()
        message(WARNING "CUDA not found - building CPU-only version")
        set(DREAMS_ENABLE_GPU OFF)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find RNS-CUDA library (optional, we include our own RNS implementation)
# find_package(rns_cuda QUIET)

# Source files
set(DREAMS_SOURCES
    src/cuda/persistent_kernel.cu
)

# Create library
if(DREAMS_ENABLE_GPU)
    add_library(dreams_lib STATIC ${DREAMS_SOURCES})
    set_target_properties(dreams_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;90"
    )
else()
    # CPU-only build
    set_source_files_properties(${DREAMS_SOURCES} PROPERTIES LANGUAGE CXX)
    add_library(dreams_lib STATIC ${DREAMS_SOURCES})
endif()

# Tests
if(DREAMS_ENABLE_TESTS)
    enable_testing()
    
    add_executable(test_persistent tests/test_persistent.cpp)
    target_link_libraries(test_persistent dreams_lib)
    if(DREAMS_ENABLE_GPU)
        set_target_properties(test_persistent PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    add_test(NAME test_persistent COMMAND test_persistent)
endif()

# Configuration summary
message(STATUS "")
message(STATUS "=== Dreams-RNS-CUDA Configuration ===")
message(STATUS "  GPU support:    ${DREAMS_ENABLE_GPU}")
message(STATUS "  Tests:          ${DREAMS_ENABLE_TESTS}")
message(STATUS "")
