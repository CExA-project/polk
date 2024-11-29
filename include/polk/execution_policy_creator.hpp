#ifndef __CREATION_POLICY_CREATOR_HPP__
#define __CREATION_POLICY_CREATOR_HPP__

#include <Kokkos_Core.hpp>

#include "kokkos_concepts.hpp"

/**
 * Polk objects.
 */
namespace polk {

/**
 * Range class.
 * Can be single-dimensional or multidimensional, but everything is treated as
 * multidimensional.
 */
template <int rank = 1> struct Range {
  static int constexpr mRank = rank;

  Kokkos::Array<std::size_t, rank> mBegin;
  Kokkos::Array<std::size_t, rank> mEnd;

public:
  /**
   * Marker to identify the class as a range.
   */
  using RangeType = Range<rank>;

  Range() = delete;

  /**
   * Multidimensional constructor.
   * @tparam rank Rank of the range.
   * @param begin Array of begin coordinates. Must have the same rank as
   * `rank`.
   * @param end Array of end coordinates. Must have the same rank as `rank`.
   */
  constexpr Range(Kokkos::Array<std::size_t, rank> begin,
                  Kokkos::Array<std::size_t, rank> end)
      : mBegin(begin), mEnd(end) {}

  /**
   * Single-dimensional constructor.
   * @param begin Begin index.
   * @param end End index.
   */
  constexpr Range(std::size_t begin, std::size_t end)
      : mBegin({begin}), mEnd({end}) {}

  /**
   * Getter for the array containing begin coordinates.
   * @return Array of coordinates.
   */
  auto constexpr getBegin() const { return mBegin; }

  /**
   * Getter for the array containing end coordinates.
   * @return Array of coordinates.
   */
  auto constexpr getEnd() const { return mEnd; }

  /**
   * Getter for the rank.
   * @return Rank of the range.
   */
  static int constexpr getRank() { return mRank; }
};

/**
 * Concept for the range.
 */
template <typename T>
concept RangeType = std::same_as<T, typename T::RangeType>;

/**
 * Tile class.
 * Can be single-dimensional (chunk size) or multidimensional, but everything
 * is treated as multidimensional.
 */
template <int rank = 1> struct Tiling {
  static int constexpr mRank = rank;
  Kokkos::Array<std::size_t, rank> mTile;

public:
  /**
   * Marker to identify the class as a tile.
   */
  using TilingType = Tiling<rank>;

  Tiling() = delete;

  /**
   * Multidimensional constructor.
   * @tparam rank Rank of the tile.
   * @param tile Array of tile. Must have the same rank as `rank`.
   */
  constexpr Tiling(Kokkos::Array<std::size_t, rank> tile) : mTile(tile) {}

  /**
   * Single-dimensional constructor.
   * It is considered that a one-dimensional tile is a chunk.
   * @param chunk_size Chunk size.
   */
  constexpr Tiling(std::size_t chunk_size) : mTile({chunk_size}) {}

  /**
   * Getter for the tile.
   * @return Array of tile.
   */
  auto constexpr getTile() const { return mTile; }

  /**
   * Getter for the rank.
   * @return Rank of the range.
   */
  static int constexpr getRank() { return mRank; }
};

/**
 * Concept for the tile.
 */
template <typename T>
concept TilingType = std::same_as<T, typename T::TilingType>;

/**
 * Default range.
 */
struct UnknownRange {};

/**
 * Default tile.
 */
struct UnknownTiling {};

/**
 * Default execution space.
 */
struct UnknownExecutionSpace {};

/**
 * Default rank.
 */
int constexpr unknownRank = 0;

/**
 * Kokkos execution policy creator.
 */
template <typename Range = UnknownRange, typename Tiling = UnknownTiling,
          typename ExecutionSpace = UnknownExecutionSpace>
class ExecutionPolicyCreator {
  Range mRange;
  Tiling mTiling;
  ExecutionSpace mExecutionSpace;

public:
  /**
   * Marker to identify the class as an execution policy creator.
   */
  using ExecutionPolicyCreatorType =
      ExecutionPolicyCreator<Range, ExecutionSpace>;

  /**
   * Default constructor.
   * @note This is the preferred constructor for this class.
   */
  constexpr ExecutionPolicyCreator() = default;

  /**
   * Full constructor.
   * @tparam Range Range class.
   * @tparam Tiling Tile class.
   * @tparam ExecutionSpace Execution space class.
   * @param r Range parameter.
   * @param t Tile parameter.
   * @param es Execution space parameter.
   * @note The user should prefer to use the default constructor.
   */
  constexpr ExecutionPolicyCreator(Range const &r, Tiling const &t,
                                   ExecutionSpace const &es)
      : mRange(r), mTiling(t), mExecutionSpace(es) {}

  /**
   * Set the range parameter.
   * The rank of the entered range must be the same of the tile, if it is set
   * already.
   * @tparam RangeIn Range class.
   * @param r Range parameter.
   * @return New execution policy creator.
   * @warning This parameter cannot be set twice.
   */
  template <RangeType RangeIn> auto constexpr with(RangeIn const &r) const {
    static_assert(std::is_same_v<Range, UnknownRange>, "Range already set");
    if constexpr (!std::is_same_v<Tiling, UnknownTiling>) {
      static_assert(Tiling::getRank() == RangeIn::getRank(),
                    "Range rank and tiling rank missmatch");
    }

    return ExecutionPolicyCreator<RangeIn, Tiling, ExecutionSpace>(
        r, mTiling, mExecutionSpace);
  }

  /**
   * Set the tile parameter.
   * The rank of the entered tile must be the same of the range, if it is set
   * already.
   * @tparam TilingIn Tile class.
   * @param t Tile parameter.
   * @return New execution policy creator.
   * @warning This parameter cannot be set twice.
   */
  template <TilingType TilingIn> auto constexpr with(TilingIn const &t) const {
    static_assert(std::is_same_v<Tiling, UnknownTiling>, "Tiling already set");
    if constexpr (!std::is_same_v<Range, UnknownRange>) {
      static_assert(Range::getRank() == TilingIn::getRank(),
                    "Range rank and tiling rank missmatch");
    }

    return ExecutionPolicyCreator<Range, TilingIn, ExecutionSpace>(
        mRange, t, mExecutionSpace);
  }

  /**
   * Set the execution space parameter.
   * @tparam ExecutionSpaceIn Execution space class.
   * @param t Tile parameter.
   * @return New execution policy creator.
   * @warning This parameter cannot be set twice.
   */
  template <kokkos_addendum::SpaceType ExecutionSpaceIn>
  auto constexpr with(ExecutionSpaceIn const &es) const {
    static_assert(std::is_same_v<ExecutionSpace, UnknownExecutionSpace>,
                  "Execution space already set");

    return ExecutionPolicyCreator<Range, Tiling, ExecutionSpaceIn>(mRange,
                                                                   mTiling, es);
  }

  /**
   * Getter for the rank.
   * It first tries to retreive the rank of the range, then the rank of the
   * tile.
   * @return Rank of execution policy creator.
   */
  static int constexpr getRank() {
    if constexpr (!std::is_same_v<Range, UnknownRange>) {
      return Range::getRank();
    }

    if constexpr (!std::is_same_v<Tiling, UnknownTiling>) {
      return Tiling::getRank();
    }

    return unknownRank;
  }

  /**
   * Getter for the range.
   * @return Range parameter.
   */
  Range constexpr getRange() const { return mRange; }

  /**
   * Getter for the tile.
   * @return Tile parameter.
   */
  Tiling constexpr getTiling() const { return mTiling; }

  /**
   * Getter for the execution space.
   * @return Execution space parameter.
   */
  ExecutionSpace constexpr getExecutionSpace() const { return mExecutionSpace; }

  /**
   * Check if rank is specified.
   * @return True if rank is not `unknownRank`.
   */
  static bool constexpr hasRank() { return getRank() != unknownRank; }

  /**
   * Check if range is specified.
   * @return True if range is not `UnknownRange`.
   */
  static bool constexpr hasRange() {
    return !std::is_same_v<Range, UnknownRange>;
  }

  /**
   * Check if tiling is specified.
   * @return True if tiling is not `UnknownTiling`.
   */
  static bool constexpr hasTiling() {
    return !std::is_same_v<Tiling, UnknownTiling>;
  }

  /**
   * Check if execution space is specified.
   * @return True if execution space is not `UnknownExecutionSpace`.
   */
  static bool constexpr hasExecutionSpace() {
    return !std::is_same_v<ExecutionSpace, UnknownExecutionSpace>;
  }

  /**
   * Retrieve a Kokkos execution policy.
   * @return Kokkos execution policy. May be a `Kokkos::RangePolicy` for
   * single-dimensional range and tile, or a `Kokkos::MDRangePolicy` for
   * multidimensional ones.
   * @warning The range (and the rank) must have been set before calling this
   * method.
   */
  auto constexpr getPolicy() const {
    // parameters that must be set
    static_assert(hasRank(), "No rank set");
    static_assert(hasRange(), "No range set");

    if constexpr (getRank() > 1) {
      if constexpr (std::is_same_v<ExecutionSpace, UnknownExecutionSpace>) {
        if constexpr (std::is_same_v<Tiling, UnknownTiling>) {
          return Kokkos::MDRangePolicy(mRange.getBegin(), mRange.getEnd());
        } else {
          return Kokkos::MDRangePolicy(mRange.getBegin(), mRange.getEnd(),
                                       mTiling.getTile());
        }
      } else {
        if constexpr (std::is_same_v<Tiling, UnknownTiling>) {
          return Kokkos::MDRangePolicy(mExecutionSpace, mRange.getBegin(),
                                       mRange.getEnd());
        } else {
          return Kokkos::MDRangePolicy(mExecutionSpace, mRange.getBegin(),
                                       mRange.getEnd(), mTiling.getTile());
        }
      }
    } else {
      if constexpr (std::is_same_v<ExecutionSpace, UnknownExecutionSpace>) {
        if constexpr (std::is_same_v<Tiling, UnknownTiling>) {
          return Kokkos::RangePolicy(mRange.getBegin()[0], mRange.getEnd()[0]);
        } else {
          return Kokkos::RangePolicy(mRange.getBegin()[0], mRange.getEnd()[0],
                                     mTiling.getTile()[0]);
        }
      } else {
        if constexpr (std::is_same_v<Tiling, UnknownTiling>) {
          return Kokkos::RangePolicy(mExecutionSpace, mRange.getBegin()[0],
                                     mRange.getEnd()[0]);
        } else {
          return Kokkos::RangePolicy(mExecutionSpace, mRange.getBegin()[0],
                                     mRange.getEnd()[0], mTiling.getTile()[0]);
        }
      }
    }
  }
};

/**
 * Concept for the execution policy creator.
 */
template <typename T>
concept ExecutionPolicyCreatorType =
    std::same_as<T, typename T::ExecutionPolicyCreatorType>;

} // namespace polk

#endif // ifndef __CREATION_POLICY_CREATOR_HPP__
