# Execution policy creator for Kokkos

Execution policy creator for Kokkos (or "polk" for short) is a header-only library that proposes a class to build an execution policy step by step.

This library is a proof of concept.

## Install

### With CMake

The best way is to use CMake.

#### As a subdirectory

Get the library in your project:

```sh
git clone https://github.com/cexa-project/polk.git path/to/polk
```

In your main CMake file:

```cmake
add_subdirectory(path/to/polk)

target_link_libraries(
    my-lib
    PRIVATE
        Polk::polk
)
```

### With FetchContent

In your main CMake file:

<!-- URL https://github.com/CExA-project/polk/archive/refs/tags/0.1.0.tar.gz -->

```cmake
include(FetchContent)
FetchContent_Declare(
    polk
    GIT_REPOSITORY https://github.com/CExA-project/polk.git
    GIT_TAG master
)
FetchContent_MakeAvailable(polk)

target_link_libraries(
    my-lib
    PRIVATE
        Polk::polk
)
```

#### As a locally available dependency

Get, then install the project:

```sh
git clone https://github.com/cexa-project/polk.git
cd polk
cmake -B build -DCMAKE_INSTALL_PREFIX=path/to/install -DCMAKE_BUILD_TYPE=Release # other Kokkos options here if needed
cmake --install build
```

In your main CMake file:

```cmake
find_package(Polk REQUIRED)

target_link_libraries(
    my-lib
    PRIVATE
        Polk::polk
)
```

### Copy files

Alternatively, you can also copy `include/polk` in your project and start using it.

## Tests

You can build tests with the CMake option `POLK_ENABLE_TESTS`, and run them with `ctest`.

If you don't have a GPU available when compiling, you have to disable the CMake option `POLK_ENABLE_GTEST_DISCOVER_TESTS`.

## Examples

You can build examples with the CMake option `POLK_ENABLE_EXAMPLES`.
They should be run individually.

## Use

The library provides a `ExecutionPolicyCreator` class that is created without arguments, and where execution policy parameters are added successively with the `with` method.
Finally, the execution policy is retrieved with the `getPolicy` method:

```cpp
#include <Kokkos_Core.hpp>
#include <polk/execution_policy_creator.hpp>

void doSomething() {
    Kokkos::parallel_for(
        "do something",
        polk::ExecutionPolicyCreator()
            .with(polk::Range<2>({0, 0}, {100, 100}))
            .with(polk::Tiling<2>({10, 10}))
            .with(Kokkos::DefaultHostExecutionSpace())
            .getPolicy(),
        KOKKOS_LAMBDA (int const i, int const j) {
            /* ... */
        }
    );
}
```

The `with` method can be called at different times:

```cpp
#include <Kokkos_Core.hpp>
#include <polk/execution_policy_creator.hpp>

template <typename ExecutionPolicyCreator, typename Kernel>
void doSomething(ExecutionPolicyCreator const& policy, Kernel const& kernel) {
    Kokkos::parallel_for(
        "do something",
        policy
            .with(Kokkos::DefaultHostExecutionSpace())
            .getPolicy(),
        kernel
    );
}

void prepareSomething() {
    doSomething(
        polk::ExecutionPolicyCreator()
            .with(polk::Range<2>({0, 0}, {100, 100}))
            .with(polk::Tiling<2>({10, 10})),
        KOKKOS_LAMBDA (int const i, int const j) {
            /* ... */
        }
    );
}
```
