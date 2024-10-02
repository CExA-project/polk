#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include "polk/execution_policy_creator.hpp"

namespace dynk {

template <typename ExecutionPolicyCreator, typename Kernel,
          typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
          typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace>
void parallel_for(bool const isExecutedOnDevice, std::string const &label,
                  ExecutionPolicyCreator const &policyCreator,
                  Kernel const &kernel) {
  Kokkos::fence("begin dynamic parallel for");
  if (isExecutedOnDevice) {
    Kokkos::parallel_for(
        label, policyCreator.with(DeviceExecutionSpace()).getPolicy(), kernel);
  } else {
    Kokkos::parallel_for(
        label, policyCreator.with(HostExecutionSpace()).getPolicy(), kernel);
  }
  Kokkos::fence("end dynamic parallel for");
}

template <
    typename T, typename... P,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
Kokkos::View<T, Kokkos::AnonymousSpace, P...>
getViewAnonymous(Kokkos::DualView<T, P...> &dualView,
                 bool const isExecutedOnDevice) {
  if (isExecutedOnDevice) {
    return dualView.template view<DeviceMemorySpace>();
  } else {
    return dualView.template view<HostMemorySpace>();
  }
}

template <
    typename DualView,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
void setModified(DualView &dualView, bool const isExecutedOnDevice) {
  if (isExecutedOnDevice) {
    dualView.template modify<DeviceMemorySpace>();
  } else {
    dualView.template modify<HostMemorySpace>();
  }
}

} // namespace dynk

int main() {
  Kokkos::ScopeGuard kokkos;

  bool const isExecutedOnDevice = true;
  Kokkos::DualView<int **> data("data", 100, 100);

  auto dataV = dynk::getViewAnonymous(data, isExecutedOnDevice);
  dynk::parallel_for(
      isExecutedOnDevice, "perform computation",
      polk::ExecutionPolicyCreator().with(polk::Range<2>({0, 0}, {100, 100})),
      KOKKOS_LAMBDA(std::size_t const i, std::size_t const j) {
        dataV(i, j) = i + j;
      });
  dynk::setModified(data, isExecutedOnDevice);

  data.template sync<Kokkos::DefaultHostExecutionSpace>();
  Kokkos::printf("Value at 50, 50: %i (should be 100)\n", data.h_view(50, 50));
}
