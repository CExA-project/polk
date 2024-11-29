#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include "polk/execution_policy_creator.hpp"

void benchmarkCreatePolkPolicy(benchmark::State &state) {
  while (state.KeepRunning()) {
    [[maybe_unused]] auto policy =
        polk::ExecutionParameters()
            .with(Kokkos::DefaultExecutionSpace{})
            .with(polk::Range<3>({0, 0, 0}, {100, 100, 100}))
            .with(polk::Tiling<3>({10, 10, 10}))
            .getPolicy();
  }
}

BENCHMARK(benchmarkCreatePolkPolicy);

void benchmarkCreateKokkosPolicy(benchmark::State &state) {
  while (state.KeepRunning()) {
    [[maybe_unused]] auto policy =
        Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace{}, {0, 0, 0},
                              {100, 100, 100}, {10, 10, 10});
  }
}

BENCHMARK(benchmarkCreateKokkosPolicy);
