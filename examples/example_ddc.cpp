#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include "polk/execution_policy_creator.hpp"

namespace ddc {

template <typename... Dim> class Domain {
  static std::size_t constexpr mRank = sizeof...(Dim);
  std::pair<Kokkos::Array<std::size_t, sizeof...(Dim)>,
            Kokkos::Array<std::size_t, sizeof...(Dim)>>
      mDimensions;

public:
  Domain(Kokkos::Array<std::size_t, sizeof...(Dim)> const &begin,
         Kokkos::Array<std::size_t, sizeof...(Dim)> const &end)
      : mDimensions(begin, end) {}

  auto begin() const { return mDimensions.first; }

  auto end() const { return mDimensions.second; }

  static std::size_t constexpr getRank() { return mRank; }
};

template <typename ExecutionPolicyCreator, typename Domain, typename Kernel>
void parallel_for_each(std::string const &label,
                       ExecutionPolicyCreator const &policyCreator,
                       Domain const &domain, Kernel const &kernel) {
  Kokkos::parallel_for(
      label,
      policyCreator
          .with(polk::Range<Domain::getRank()>(domain.begin(), domain.end()))
          .getPolicy(),
      kernel);
}

} // namespace ddc

struct X {};

struct Y {};

int main() {
  Kokkos::ScopeGuard kokkos;

  Kokkos::View<int **, Kokkos::HostSpace> data("data", 100, 100);
  ddc::Domain<X, Y> domain({1, 1}, {99, 99});

  ddc::parallel_for_each(
      "perform computation",
      polk::ExecutionPolicyCreator().with(Kokkos::DefaultHostExecutionSpace()),
      domain, KOKKOS_LAMBDA(std::size_t const i, std::size_t const j) {
        data(i, j) = i + j;
      });

  Kokkos::printf("Value at 50, 50: %i (should be 100)\n", data(50, 50));
}
