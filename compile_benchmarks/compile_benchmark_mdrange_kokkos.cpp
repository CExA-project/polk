#include <Kokkos_Core.hpp>

#include "polk/execution_policy_creator.hpp"

int main() {
  [[maybe_unused]] auto policy =
      Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace{}, {0, 0, 0},
                            {100, 100, 100}, {10, 10, 10});
}

