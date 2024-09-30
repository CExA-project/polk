add_executable(
    example-dynk
    example_dynk.cpp
)

target_link_libraries(
    example-dynk
    Polk::polk
)

add_executable(
    example-ddc
    example_ddc.cpp
)

target_link_libraries(
    example-ddc
    Polk::polk
)
