cmake_minimum_required(VERSION 3.20)

project(parabola LANGUAGES CXX)

# -----------------------------
# Platform-specific compiler selection
# -----------------------------
if(NOT DEFINED ENV{CC})
    if(APPLE)
        # macOS: Homebrew GCC oder AppleClang fallback
        set(CMAKE_C_COMPILER "/opt/homebrew/bin/gcc" CACHE STRING "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++" CACHE STRING "C++ compiler" FORCE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    elseif(UNIX)
        # Linux
        set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE STRING "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE STRING "C++ compiler" FORCE)
    endif()
endif()

# -----------------------------
# C++ standard and optimization
# -----------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -fvisibility=hidden -Wall -Wextra)
endif()

# -----------------------------
# OpenMP detection
# -----------------------------
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
    add_compile_options(${OpenMP_CXX_FLAGS})
    set(OPENMP_LIBS ${OpenMP_CXX_LIBRARIES})
else()
    message(WARNING "OpenMP not found. Parallelization disabled.")
    set(OPENMP_LIBS "")
endif()

# -----------------------------
# pybind11
# -----------------------------
find_package(pybind11 CONFIG REQUIRED)

# -----------------------------
# Build the extension
# -----------------------------
pybind11_add_module(parabola_ext ./CPP_Extension/src/AtomicBasis.cpp)

# OpenMP libraries
target_link_libraries(parabola_ext PRIVATE ${OPENMP_LIBS})

# Output name for Python
set_target_properties(parabola_ext PROPERTIES OUTPUT_NAME "_extension")

# Install
install(TARGETS parabola_ext DESTINATION parabola)

