#include <Kokkos_Core.hpp>

#include "polk/execution_policy_creator.hpp"

int main() {
  [[maybe_unused]] auto policy = polk::ExecutionPolicyCreator()
                                     .with(Kokkos::DefaultExecutionSpace{})
                                     .with(polk::Range(0, 100))
                                     .with(polk::Tiling(10))
                                     .getPolicy();
}
