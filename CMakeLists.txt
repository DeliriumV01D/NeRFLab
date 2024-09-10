cmake_minimum_required (VERSION 3.8)

project(NeRFLab)

find_package(Torch REQUIRED)
find_package(OpenCV 4.6 REQUIRED)
find_package(Qt6 REQUIRED Core)
find_package(Qt6 REQUIRED Widgets)
find_package(Qt6 REQUIRED Gui)
find_package(OpenMP)
# find_package(CUDA)
# find_package(CUDAToolkit REQUIRED)
#find_package(faiss REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 20)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

include("CMakeLists.Files.txt")

include_directories(${CMAKE_SOURCE_DIR}/src)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)


if(MSVC_IDE)
	source_group("src" FILES ${Files_src})

	source_group("" FILES CMakeLists.Files.txt)
endif()

if(NOT HEADERS STREQUAL "")
	qt6_wrap_cpp(MOC_SOURCES ${HEADERS})
	source_group("_moc" FILES ${MOC_SOURCES})
	set(SOURCES ${SOURCES} ${MOC_SOURCES})
endif()

source_group("Headers" FILES ${HEADERS})
set(SOURCES ${SOURCES} ${HEADERS})

if(NOT QtResources STREQUAL "")
	qt6_add_resources(QRC_SOURCES ${QtResources})
	if(MSVC_IDE)
		source_group("_qrc" FILES ${QRC_SOURCES})
	endif()
	set(SOURCES ${SOURCES} ${QRC_SOURCES})
endif()

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS} PRIVATE ${LINK_LIBS})

if (MSVC) 
add_definitions(-D_DISABLE_EXTENDED_ALIGNED_STORAGE)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET NeRFLab
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:NeRFLab>)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /openmp")
endif()
