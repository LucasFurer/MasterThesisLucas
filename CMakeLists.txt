cmake_minimum_required(VERSION 3.10)
project(MasterThesisLucas VERSION 0.1.0)

cmake_policy(SET CMP0072 NEW)
find_package(OpenGL REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB IMGUI_SOURCES
    src/library_src/include/imgui/imgui.cpp
    src/library_src/include/imgui/imgui_draw.cpp
    src/library_src/include/imgui/imgui_tables.cpp
    src/library_src/include/imgui/imgui_widgets.cpp
    src/library_src/include/imgui/imgui_demo.cpp
    src/library_src/include/imgui/imgui_impl_glfw.cpp
    src/library_src/include/imgui/imgui_impl_opengl3.cpp
)

file(GLOB GLAD_SOURCES 
    src/library_src/include/glad/glad.h 
    src/library_src/include/glad/glad.c
)

file(GLOB CUSTOM_SOURCES
    src/nbodysolvers/*.h
    src/particles/*.h
    src/trees/*.h
    src/*.h
    src/*.cpp
)

add_executable(MasterThesisLucas
    src/main.cpp

    ${CUSTOM_SOURCES}

    ${GLAD_SOURCES}

    ${IMGUI_SOURCES}
)

target_link_libraries(
    ${PROJECT_NAME} 
    glfw
    OpenGL::GL
    fftw3
    #eigen3
)


include_directories(
    src/library_src/include
)