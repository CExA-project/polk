#ifndef __POLK_KOKKOS_CONCEPTS_HPP__
#define __POLK_KOKKOS_CONCEPTS_HPP__

namespace kokkos_addendum {

/**
 * Space trait value.
 */
template <typename T> bool constexpr is_space_v = Kokkos::is_space<T>::value;

/**
 * Space concept.
 */
template <typename T>
concept SpaceType = is_space_v<T>;

} // namespace kokkos_addendum

#endif // ifndef __POLK_KOKKOS_CONCEPTS_HPP__
