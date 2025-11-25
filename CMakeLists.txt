cmake_minimum_required(VERSION 3.17)
project(MA-CETSP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# ------------ Common Includes --------------
include_directories("include/")
include_directories("include/Genetic/")
include_directories("include/Genetic/Crossover/")
include_directories("include/LocalSearch/")
include_directories("include/Utils/")

# ------------ Sources --------------
aux_source_directory("src/" SRC)
aux_source_directory("src/Genetic" GENETIC)
aux_source_directory("src/Genetic/Crossover" CROSSOVER)
aux_source_directory("src/LocalSearch" LOCALSEARCH)
aux_source_directory("src/Utils" UTILS)
add_executable(MA-CETSP ${SRC} ${GENETIC} ${CROSSOVER} ${LOCALSEARCH} ${UTILS})

# ==========================================================
#                 WINDOWS (MSVC) CONFIG
# ==========================================================
if (MSVC)
    message(STATUS "🟦 Configuring for Windows / MSVC...")

    # Gurobi installation path (adjust if needed)
    set(GUROBI_HOME "C:/gurobi1203/win64")
    include_directories("${GUROBI_HOME}/include")
    link_directories("${GUROBI_HOME}/lib")

    # Determine Visual Studio toolset year
    if (MSVC_TOOLSET_VERSION EQUAL 143)
        set(MSVC_YEAR "2022")
    elseif(MSVC_TOOLSET_VERSION EQUAL 142)
        set(MSVC_YEAR "2019")
    elseif(MSVC_TOOLSET_VERSION EQUAL 141)
        set(MSVC_YEAR "2017")
    else()
        set(MSVC_YEAR "2017")
    endif()

    # Prefer multi-threaded DLL runtime
    set(M_FLAG "md")

    # Try to find the correct Gurobi C++ lib for this toolset
    find_library(GUROBI_CXX_LIBRARY
        NAMES gurobi_c++${M_FLAG}${MSVC_YEAR} gurobi_c++${M_FLAG}2017
        HINTS "${GUROBI_HOME}/lib"
    )

    if(NOT GUROBI_CXX_LIBRARY)
        message(WARNING "⚠️  Could not find specific Gurobi C++ library for toolset, falling back to gurobi_c++md2017.lib")
        set(GUROBI_CXX_LIBRARY "${GUROBI_HOME}/lib/gurobi_c++md2017.lib")
    endif()

    # C core library (required)
    set(GUROBI_C_LIBRARY "${GUROBI_HOME}/lib/gurobi120.lib")

    # Link Gurobi
    target_link_libraries(MA-CETSP
        ${GUROBI_CXX_LIBRARY}
        ${GUROBI_C_LIBRARY}
    )

    # Optional optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /EHsc /DNOMINMAX")
    message(STATUS "✅ Linked Gurobi libs: ${GUROBI_CXX_LIBRARY}, ${GUROBI_C_LIBRARY}")

# ==========================================================
#                 LINUX (GCC) CONFIG
# ==========================================================
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "🟩 Configuring for Linux / GCC...")

    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")

    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    find_package(GUROBI REQUIRED)
    set(GUROBI_CXX_LIBRARY "$ENV{GUROBI_HOME}/lib/libgurobi_c++.a")

    include_directories(${GUROBI_INCLUDE_DIRS})

    target_link_libraries(MA-CETSP
        ${GUROBI_CXX_LIBRARY}
        ${GUROBI_LIBRARY}
    )

    message(STATUS "✅ Linked Gurobi libs (Linux): ${GUROBI_CXX_LIBRARY}, ${GUROBI_LIBRARY}")
endif()

message(STATUS "✅ CMake configuration successful for ${CMAKE_SYSTEM_NAME}")
