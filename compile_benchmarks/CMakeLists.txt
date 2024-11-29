add_library(
    compile-benchmark-range-polk
    OBJECT
    compile_benchmark_range_polk.cpp
)

target_link_libraries(
    compile-benchmark-range-polk
    Polk::polk
)

add_library(
    compile-benchmark-range-kokkos
    OBJECT
    compile_benchmark_range_kokkos.cpp
)

target_link_libraries(
    compile-benchmark-range-kokkos
    Polk::polk
)

add_library(
    compile-benchmark-mdrange-polk
    OBJECT
    compile_benchmark_mdrange_polk.cpp
)

target_link_libraries(
    compile-benchmark-mdrange-polk
    Polk::polk
)

add_library(
    compile-benchmark-mdrange-kokkos
    OBJECT
    compile_benchmark_mdrange_kokkos.cpp
)

target_link_libraries(
    compile-benchmark-mdrange-kokkos
    Polk::polk
)
