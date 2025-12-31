cmake_minimum_required(VERSION 3.15)
project(DecompressorProject LANGUAGES CXX)

# --- 1. C++ Standard Setup ---
# SZ3 typically requires C++17 or newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 2. Set Search Paths ---
# Point CMake to your specific SZ3 install location
set(SZ3_INSTALL_PATH "/home/szhang/SZ3/SZ3-for-nekRS/install")
set(VTK_DIR "/home/szhang/VTK/build-with-SZ3/lib/cmake/vtk-9.5")
list(APPEND CMAKE_PREFIX_PATH "${SZ3_INSTALL_PATH}")

# --- 3. Find Dependencies ---
# Find MPI
find_package(MPI REQUIRED)

# Find SZ3
# This looks for SZ3Config.cmake in the prefix path defined above
find_package(SZ3 REQUIRED)

# Find VTK
find_package(VTK REQUIRED)

# --- 4. Define Executable ---
add_executable(decompressor decompressor.cpp)

# --- 5. Link Libraries ---
target_link_libraries(decompressor
    PRIVATE
    MPI::MPI_CXX    # Links MPI headers and libraries
    SZ3::SZ3        # Links SZ3 headers and libraries (Modern CMake target)
    ${VTK_LIBRARIES}
)

# Initialize VTK modules (required for some versions/modules of VTK to function correctly)
vtk_module_autoinit(
    TARGETS decompressor
    MODULES ${VTK_LIBRARIES}
)

# Optional: Add standard compiler warnings to catch issues early
if(NOT MSVC)
    target_compile_options(decompressor PRIVATE -Wall -Wextra)
endif()