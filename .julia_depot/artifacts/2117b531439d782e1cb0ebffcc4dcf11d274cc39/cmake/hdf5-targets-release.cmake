#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hdf5-shared" for configuration "Release"
set_property(TARGET hdf5-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5-shared PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "libaec::sz;libaec::aec"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5.so.310.5.1"
  IMPORTED_SONAME_RELEASE "libhdf5.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5-shared "${_IMPORT_PREFIX}/lib/libhdf5.so.310.5.1" )

# Import target "mirror_server" for configuration "Release"
set_property(TARGET mirror_server APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mirror_server PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mirror_server"
  )

list(APPEND _IMPORT_CHECK_TARGETS mirror_server )
list(APPEND _IMPORT_CHECK_FILES_FOR_mirror_server "${_IMPORT_PREFIX}/bin/mirror_server" )

# Import target "mirror_server_stop" for configuration "Release"
set_property(TARGET mirror_server_stop APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mirror_server_stop PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mirror_server_stop"
  )

list(APPEND _IMPORT_CHECK_TARGETS mirror_server_stop )
list(APPEND _IMPORT_CHECK_FILES_FOR_mirror_server_stop "${_IMPORT_PREFIX}/bin/mirror_server_stop" )

# Import target "hdf5_tools-shared" for configuration "Release"
set_property(TARGET hdf5_tools-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_tools-shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5_tools.so.310.0.6"
  IMPORTED_SONAME_RELEASE "libhdf5_tools.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_tools-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_tools-shared "${_IMPORT_PREFIX}/lib/libhdf5_tools.so.310.0.6" )

# Import target "h5diff" for configuration "Release"
set_property(TARGET h5diff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5diff PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5diff"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5diff )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5diff "${_IMPORT_PREFIX}/bin/h5diff" )

# Import target "ph5diff" for configuration "Release"
set_property(TARGET ph5diff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ph5diff PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ph5diff"
  )

list(APPEND _IMPORT_CHECK_TARGETS ph5diff )
list(APPEND _IMPORT_CHECK_FILES_FOR_ph5diff "${_IMPORT_PREFIX}/bin/ph5diff" )

# Import target "h5ls" for configuration "Release"
set_property(TARGET h5ls APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5ls PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5ls"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5ls )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5ls "${_IMPORT_PREFIX}/bin/h5ls" )

# Import target "h5debug" for configuration "Release"
set_property(TARGET h5debug APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5debug PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5debug"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5debug )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5debug "${_IMPORT_PREFIX}/bin/h5debug" )

# Import target "h5repart" for configuration "Release"
set_property(TARGET h5repart APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5repart PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5repart"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5repart )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5repart "${_IMPORT_PREFIX}/bin/h5repart" )

# Import target "h5mkgrp" for configuration "Release"
set_property(TARGET h5mkgrp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5mkgrp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5mkgrp"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5mkgrp )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5mkgrp "${_IMPORT_PREFIX}/bin/h5mkgrp" )

# Import target "h5clear" for configuration "Release"
set_property(TARGET h5clear APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5clear PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5clear"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5clear )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5clear "${_IMPORT_PREFIX}/bin/h5clear" )

# Import target "h5delete" for configuration "Release"
set_property(TARGET h5delete APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5delete PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5delete"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5delete )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5delete "${_IMPORT_PREFIX}/bin/h5delete" )

# Import target "h5import" for configuration "Release"
set_property(TARGET h5import APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5import PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5import"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5import )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5import "${_IMPORT_PREFIX}/bin/h5import" )

# Import target "h5repack" for configuration "Release"
set_property(TARGET h5repack APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5repack PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5repack"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5repack )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5repack "${_IMPORT_PREFIX}/bin/h5repack" )

# Import target "h5jam" for configuration "Release"
set_property(TARGET h5jam APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5jam PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5jam"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5jam )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5jam "${_IMPORT_PREFIX}/bin/h5jam" )

# Import target "h5unjam" for configuration "Release"
set_property(TARGET h5unjam APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5unjam PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5unjam"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5unjam )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5unjam "${_IMPORT_PREFIX}/bin/h5unjam" )

# Import target "h5copy" for configuration "Release"
set_property(TARGET h5copy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5copy PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5copy"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5copy )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5copy "${_IMPORT_PREFIX}/bin/h5copy" )

# Import target "h5stat" for configuration "Release"
set_property(TARGET h5stat APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5stat PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5stat"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5stat )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5stat "${_IMPORT_PREFIX}/bin/h5stat" )

# Import target "h5dump" for configuration "Release"
set_property(TARGET h5dump APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5dump PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5dump"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5dump )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5dump "${_IMPORT_PREFIX}/bin/h5dump" )

# Import target "h5format_convert" for configuration "Release"
set_property(TARGET h5format_convert APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5format_convert PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5format_convert"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5format_convert )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5format_convert "${_IMPORT_PREFIX}/bin/h5format_convert" )

# Import target "h5perf_serial" for configuration "Release"
set_property(TARGET h5perf_serial APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5perf_serial PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5perf_serial"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5perf_serial )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5perf_serial "${_IMPORT_PREFIX}/bin/h5perf_serial" )

# Import target "h5perf" for configuration "Release"
set_property(TARGET h5perf APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5perf PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5perf"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5perf )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5perf "${_IMPORT_PREFIX}/bin/h5perf" )

# Import target "hdf5_hl-shared" for configuration "Release"
set_property(TARGET hdf5_hl-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_hl-shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5_hl.so.310.0.6"
  IMPORTED_SONAME_RELEASE "libhdf5_hl.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl-shared "${_IMPORT_PREFIX}/lib/libhdf5_hl.so.310.0.6" )

# Import target "h5watch" for configuration "Release"
set_property(TARGET h5watch APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(h5watch PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/h5watch"
  )

list(APPEND _IMPORT_CHECK_TARGETS h5watch )
list(APPEND _IMPORT_CHECK_FILES_FOR_h5watch "${_IMPORT_PREFIX}/bin/h5watch" )

# Import target "hdf5_f90cstub-shared" for configuration "Release"
set_property(TARGET hdf5_f90cstub-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_f90cstub-shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5_f90cstub.so.310.3.2"
  IMPORTED_SONAME_RELEASE "libhdf5_f90cstub.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_f90cstub-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_f90cstub-shared "${_IMPORT_PREFIX}/lib/libhdf5_f90cstub.so.310.3.2" )

# Import target "hdf5_fortran-shared" for configuration "Release"
set_property(TARGET hdf5_fortran-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_fortran-shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5_fortran.so.310.3.2"
  IMPORTED_SONAME_RELEASE "libhdf5_fortran.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_fortran-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_fortran-shared "${_IMPORT_PREFIX}/lib/libhdf5_fortran.so.310.3.2" )

# Import target "hdf5_hl_f90cstub-shared" for configuration "Release"
set_property(TARGET hdf5_hl_f90cstub-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_hl_f90cstub-shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5_hl_f90cstub.so.310.0.6"
  IMPORTED_SONAME_RELEASE "libhdf5_hl_f90cstub.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl_f90cstub-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl_f90cstub-shared "${_IMPORT_PREFIX}/lib/libhdf5_hl_f90cstub.so.310.0.6" )

# Import target "hdf5_hl_fortran-shared" for configuration "Release"
set_property(TARGET hdf5_hl_fortran-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_hl_fortran-shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5_hl_fortran.so.310.0.6"
  IMPORTED_SONAME_RELEASE "libhdf5_hl_fortran.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl_fortran-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl_fortran-shared "${_IMPORT_PREFIX}/lib/libhdf5_hl_fortran.so.310.0.6" )

# Import target "hdf5_cpp-shared" for configuration "Release"
set_property(TARGET hdf5_cpp-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_cpp-shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5_cpp.so.310.0.6"
  IMPORTED_SONAME_RELEASE "libhdf5_cpp.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_cpp-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_cpp-shared "${_IMPORT_PREFIX}/lib/libhdf5_cpp.so.310.0.6" )

# Import target "hdf5_hl_cpp-shared" for configuration "Release"
set_property(TARGET hdf5_hl_cpp-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_hl_cpp-shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf5_hl_cpp.so.310.0.6"
  IMPORTED_SONAME_RELEASE "libhdf5_hl_cpp.so.310"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl_cpp-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl_cpp-shared "${_IMPORT_PREFIX}/lib/libhdf5_hl_cpp.so.310.0.6" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
