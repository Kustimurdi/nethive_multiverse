#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "chemfiles" for configuration "Release"
set_property(TARGET chemfiles APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(chemfiles PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libchemfiles.so.0.10.4"
  IMPORTED_SONAME_RELEASE "libchemfiles.so.0.10"
  )

list(APPEND _IMPORT_CHECK_TARGETS chemfiles )
list(APPEND _IMPORT_CHECK_FILES_FOR_chemfiles "${_IMPORT_PREFIX}/lib/libchemfiles.so.0.10.4" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
