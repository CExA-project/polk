add_executable(
    benchmark-range
    benchmark_range.cpp
)

target_link_libraries(
    benchmark-range
    benchmark::benchmark
    Polk::polk
)

add_executable(
    benchmark-mdrange
    benchmark_mdrange.cpp
)

target_link_libraries(
    benchmark-mdrange
    benchmark::benchmark
    Polk::polk
)
