#include <Kokkos_Core.hpp>

#include "polk/execution_policy_creator.hpp"

int main() {
  [[maybe_unused]] auto policy =
      polk::ExecutionPolicyCreator()
          .with(Kokkos::DefaultExecutionSpace{})
          .with(polk::Range<3>({0, 0, 0}, {100, 100, 100}))
          .with(polk::Tiling<3>({10, 10, 10}))
          .getPolicy();
}
