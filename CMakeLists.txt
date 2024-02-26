cmake_minimum_required (VERSION 3.8)
project (simplerf)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 17)
option(USE_VULKAN "Enable GPU implementation via Vulkan" OFF)

option(USE_ENZYME "Enable Enzyme AD" OFF)

if(USE_ENZYME)
  set(CLANG_VERSION "17" CACHE STRING "clang and LLVM version to use")
  set(CMAKE_C_COMPILER   "/usr/bin/clang-${CLANG_VERSION}")   # $ export CC=/usr/bin/clang-17
  set(CMAKE_CXX_COMPILER "/usr/bin/clang++-${CLANG_VERSION}") # $ export CXX=/usr/bin/clang++-17
  set(ENZYME_PLUGIN_DLL  "/home/egorf/LiteRF/enzyme/build/Enzyme/ClangEnzyme-${CLANG_VERSION}.so")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  add_definitions(-DUSE_ENZYME)
endif()

find_package(OpenMP)

if(USE_VULKAN)
  
  find_package(Vulkan)

  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
  elseif(CMAKE_SYSTEM_NAME STREQUAL Linux)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_XLIB_KHR)
  elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_MACOS_MVK)
  endif()

  add_subdirectory(external/volk)

  add_compile_definitions(USE_VOLK)
  add_compile_definitions(USE_VULKAN)
  
  include_directories(${Vulkan_INCLUDE_DIR})

  include_directories("external/LiteMath"
                      "external/vkutils"
                      "external/volk")
                    
  link_directories("../volk")
else()
  include_directories("external/LiteMath")
endif()

include_directories("./external/nlohmann")
include_directories("./external")
include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -O0 -fopenmp")

if(USE_VULKAN)
  add_executable(testapp main.cpp 
                 example_tracer/example_tracer.cpp
                 external/LiteMath/Image2d.cpp 
                 
                 example_tracer/example_tracer_generated.cpp
                 example_tracer/example_tracer_generated_ds.cpp
                 example_tracer/example_tracer_generated_init.cpp  

                 external/vkutils/vk_utils.cpp
                 external/vkutils/vk_copy.cpp
                 external/vkutils/vk_context.cpp
                 external/vkutils/vk_alloc_simple.cpp
                 external/vkutils/vk_buffers.cpp
                 external/vkutils/vk_images.cpp
                 external/vkutils/vk_pipeline.cpp
                 external/vkutils/vk_descriptor_sets.cpp)

  set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  target_link_libraries(testapp LINK_PUBLIC OpenMP::OpenMP_CXX ${Vulkan_LIBRARY} volk dl)

else()
  add_executable(testapp main.cpp 
                 example_tracer/example_tracer.cpp
                 external/LiteMath/Image2d.cpp) 


if(USE_ENZYME)
  set(ENZYME_FLAGS "-fplugin=${ENZYME_PLUGIN_DLL}") # -fplugin=${ENZYME_PLUGIN_DLL}
  target_compile_options(testapp PUBLIC ${OMP_FLAGS} ${ENZYME_FLAGS})
endif()

  set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

  target_link_libraries(testapp LINK_PUBLIC)
endif()

