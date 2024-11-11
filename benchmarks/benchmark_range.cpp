#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include "polk/execution_policy_creator.hpp"

void benchmarkCreatePolkPolicy(benchmark::State &state) {
  while (state.KeepRunning()) {
    [[maybe_unused]] auto policy = polk::ExecutionPolicyCreator()
                                       .with(Kokkos::DefaultExecutionSpace{})
                                       .with(polk::Range(0, 100))
                                       .with(polk::Tiling(10))
                                       .getPolicy();
  }
}

BENCHMARK(benchmarkCreatePolkPolicy);

void benchmarkCreateKokkosPolicy(benchmark::State &state) {
  while (state.KeepRunning()) {
    [[maybe_unused]] auto policy =
        Kokkos::RangePolicy(Kokkos::DefaultExecutionSpace{}, 0, 100, 10);
  }
}

BENCHMARK(benchmarkCreateKokkosPolicy);
