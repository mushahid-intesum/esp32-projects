cmake_minimum_required(VERSION 3.16)

project(
    bird_call
    VERSION 1.0
    DESCRIPTION "Bird Call Identification on ESP32"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(LIBTORCH_PATH "/mnt/Stuff/phd_projects/esp32-projects/bird_call_id/esp32_code/include/torch")
set(c10_PATH "/mnt/Stuff/phd_projects/esp32-projects/bird_call_id/esp32_code/include/c10")
set(EXECUTORCH_PATH "/mnt/Stuff/phd_projects/esp32-projects/bird_call_id/esp32_code/include/executorch")

find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})
find_package(c10 REQUIRED PATHS ${c10_PATH})
find_package(Executorch REQUIRED PATHS ${EXECUTORCH_PATH})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(
    ${PROJECT_NAME}
    src/main.cpp
)

target_link_libraries(
    bird_call 
    PRIVATE 
    "${TORCH_LIBRARIES}"
)

target_include_directories(bird_call PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    executorch
    extension_module_static
    extension_flat_tensor
    extension_tensor
    optimized_native_cpu_ops_lib
    xnnpack_backend
    c10
    torch
)
