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
include_directories("include/ML/")
include_directories("include/Features/")



# ------------ Sources --------------
aux_source_directory("src/" SRC)
aux_source_directory("src/Genetic" GENETIC)
aux_source_directory("src/Genetic/Crossover" CROSSOVER)
aux_source_directory("src/LocalSearch" LOCALSEARCH)
aux_source_directory("src/Utils" UTILS)
add_executable(MA-CETSP ${SRC} ${GENETIC} ${CROSSOVER} ${LOCALSEARCH} ${UTILS} "src/ML/SurvivalModel.cpp" "src/Features/GeometryFeatures.cpp")

# ==========================================================
#                 WINDOWS (MSVC) CONFIG
# ==========================================================
if (MSVC)
    message(STATUS "🟦 Configuring for Windows / MSVC...")

    # =====================
    # GUROBI
    # =====================
    set(GUROBI_HOME "C:/gurobi1203/win64")
    include_directories("${GUROBI_HOME}/include")

    if (MSVC_TOOLSET_VERSION EQUAL 143)
        set(MSVC_YEAR "2022")
    elseif (MSVC_TOOLSET_VERSION EQUAL 142)
        set(MSVC_YEAR "2019")
    else()
        set(MSVC_YEAR "2017")
    endif()

    set(M_FLAG "md")

    find_library(GUROBI_CXX_LIBRARY
        NAMES gurobi_c++${M_FLAG}${MSVC_YEAR} gurobi_c++${M_FLAG}2017
        HINTS "${GUROBI_HOME}/lib"
        REQUIRED
    )

    set(GUROBI_C_LIBRARY "${GUROBI_HOME}/lib/gurobi120.lib")





    # =====================
    # FINAL LINKING (ONLY ONCE)
    # =====================
    target_link_libraries(MA-CETSP PRIVATE
        ${GUROBI_CXX_LIBRARY}
        ${GUROBI_C_LIBRARY}
    )

    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /EHsc /DNOMINMAX")
endif()



message(STATUS "✅ CMake configuration successful for ${CMAKE_SYSTEM_NAME}")
