#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include "polk/execution_policy_creator.hpp"

void createPolkPolicy() {
  [[maybe_unused]] auto policy =
      polk::ExecutionPolicyCreator()
          .with(polk::Range<3>({0, 0, 0}, {100, 100, 100}))
          .with(polk::Tiling<3>({10, 10, 10}))
          .getPolicy();
}

void benchmarkCreatePolkPolicy(benchmark::State &state) {
  while (state.KeepRunning()) {
    createPolkPolicy();
  }
}

BENCHMARK(benchmarkCreatePolkPolicy);

void createKokkosPolicy() {
  [[maybe_unused]] auto policy =
      Kokkos::MDRangePolicy({0, 0, 0}, {100, 100, 100}, {10, 10, 10});
}

void benchmarkCreateKokkosPolicy(benchmark::State &state) {
  while (state.KeepRunning()) {
    createKokkosPolicy();
  }
}

BENCHMARK(benchmarkCreateKokkosPolicy);

BENCHMARK_MAIN();
