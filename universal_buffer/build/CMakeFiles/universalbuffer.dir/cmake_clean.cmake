file(REMOVE_RECURSE
  "libuniversalbuffer.pdb"
  "libuniversalbuffer.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/universalbuffer.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
