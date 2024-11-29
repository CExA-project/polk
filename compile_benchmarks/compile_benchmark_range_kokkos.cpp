#include <Kokkos_Core.hpp>

#include "polk/execution_policy_creator.hpp"

int main() {
  [[maybe_unused]] auto policy =
      Kokkos::RangePolicy(Kokkos::DefaultExecutionSpace{}, 0, 100, 10);
}
