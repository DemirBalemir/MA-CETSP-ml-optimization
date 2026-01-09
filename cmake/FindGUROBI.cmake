# --- FindGUROBI.cmake ---
# Compatible with Gurobi 12.0.3 (Windows / Linux) and Visual Studio 2017–2022

find_path(
  GUROBI_INCLUDE_DIRS
  NAMES gurobi_c.h
  HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
  PATH_SUFFIXES include)

# Look for base library (add gurobi120 for v12.0.3)
find_library(
  GUROBI_LIBRARY
  NAMES gurobi gurobi95 gurobi100 gurobi110 gurobi120
  HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
  PATH_SUFFIXES lib)

# ----------------------------------------------------
# C++ interface
# ----------------------------------------------------
if(CXX)
  if(MSVC)
    # determine Visual Studio toolset
    if(MSVC_TOOLSET_VERSION EQUAL 143)
      set(MSVC_YEAR "2022")
    elseif(MSVC_TOOLSET_VERSION EQUAL 142)
      set(MSVC_YEAR "2019")
    elseif(MSVC_TOOLSET_VERSION EQUAL 141)
      set(MSVC_YEAR "2017")
    elseif(MSVC_TOOLSET_VERSION EQUAL 140)
      set(MSVC_YEAR "2015")
    endif()

    if(MT)
      set(M_FLAG "mt")
    else()
      set(M_FLAG "md")
    endif()

    # --- Primary search ---
    find_library(
      GUROBI_CXX_LIBRARY
      NAMES gurobi_c++${M_FLAG}${MSVC_YEAR}
      HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
      PATH_SUFFIXES lib)

    find_library(
      GUROBI_CXX_DEBUG_LIBRARY
      NAMES gurobi_c++${M_FLAG}d${MSVC_YEAR}
      HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
      PATH_SUFFIXES lib)

    # --- Fallback for Gurobi 12.0.3 which still uses 2017 suffix ---
    if(NOT GUROBI_CXX_LIBRARY)
      find_library(
        GUROBI_CXX_LIBRARY
        NAMES gurobi_c++md2017 gurobi_c++mdd2017 gurobi_c++mt2017 gurobi_c++mtd2017
        HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
        PATH_SUFFIXES lib)
    endif()

  else()
    # Linux / GCC
    find_library(
      GUROBI_CXX_LIBRARY
      NAMES gurobi_c++
      HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
      PATH_SUFFIXES lib)
    set(GUROBI_CXX_DEBUG_LIBRARY ${GUROBI_CXX_LIBRARY})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG
  GUROBI_LIBRARY
  GUROBI_INCLUDE_DIRS)
