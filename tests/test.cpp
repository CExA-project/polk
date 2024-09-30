#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "polk/execution_policy_creator.hpp"

TEST(test_range, test_create) {
  auto myRange = polk::Range<2>({0, 0}, {1, 1});

  static_assert(myRange.getRank() == 2);

  ASSERT_EQ(myRange.getBegin()[0], 0);
  ASSERT_EQ(myRange.getBegin()[1], 0);
  ASSERT_EQ(myRange.getEnd()[0], 1);
  ASSERT_EQ(myRange.getEnd()[1], 1);
}

TEST(test_range, test_create_array) {
    Kokkos::Array<std::size_t, 2> begin {0, 0}, end {1, 1};
  auto myRange = polk::Range<2>(begin, end);

  static_assert(myRange.getRank() == 2);

  ASSERT_EQ(myRange.getBegin()[0], 0);
  ASSERT_EQ(myRange.getBegin()[1], 0);
  ASSERT_EQ(myRange.getEnd()[0], 1);
  ASSERT_EQ(myRange.getEnd()[1], 1);
}

TEST(test_range, test_create_array_move) {
  auto myRange = polk::Range<2>(
              Kokkos::Array<std::size_t, 2>{0, 0},
              Kokkos::Array<std::size_t, 2>{1, 1}
          );

  static_assert(myRange.getRank() == 2);

  ASSERT_EQ(myRange.getBegin()[0], 0);
  ASSERT_EQ(myRange.getBegin()[1], 0);
  ASSERT_EQ(myRange.getEnd()[0], 1);
  ASSERT_EQ(myRange.getEnd()[1], 1);
}

TEST(test_tiling, test_create) {
  auto myTiling = polk::Tiling<2>({10, 10});

  static_assert(myTiling.getRank() == 2);

  ASSERT_EQ(myTiling.getTile()[0], 10);
  ASSERT_EQ(myTiling.getTile()[1], 10);
}

TEST(test_execution_policy_creator, test_default) {
  auto myPolicyCreator = polk::ExecutionPolicyCreator();

  static_assert(myPolicyCreator.getRank() == polk::unknownRank);
  static_assert(
      std::is_same_v<decltype(myPolicyCreator.getRange()), polk::UnknownRange>);
  static_assert(std::is_same_v<decltype(myPolicyCreator.getExecutionSpace()),
                               polk::UnknownExecutionSpace>);
}

TEST(test_execution_policy_creator, test_with_range) {
  auto myRange = polk::Range<2>({0, 0}, {1, 1});
  auto myPolicyCreator = polk::ExecutionPolicyCreator().with(myRange);

  static_assert(myPolicyCreator.getRank() == 2);
  static_assert(
      std::is_same_v<decltype(myPolicyCreator.getRange()), decltype(myRange)>);
  static_assert(std::is_same_v<decltype(myPolicyCreator.getTiling()),
                               polk::UnknownTiling>);
  static_assert(std::is_same_v<decltype(myPolicyCreator.getExecutionSpace()),
                               polk::UnknownExecutionSpace>);
}

TEST(test_execution_policy_creator, test_with_tiling) {
  auto myTiling = polk::Tiling<2>({10, 10});
  auto myPolicyCreator = polk::ExecutionPolicyCreator().with(myTiling);

  static_assert(myPolicyCreator.getRank() == 2);
  static_assert(
      std::is_same_v<decltype(myPolicyCreator.getRange()), polk::UnknownRange>);
  static_assert(std::is_same_v<decltype(myPolicyCreator.getTiling()),
                               decltype(myTiling)>);
  static_assert(std::is_same_v<decltype(myPolicyCreator.getExecutionSpace()),
                               polk::UnknownExecutionSpace>);
}

TEST(test_execution_policy_creator, test_with_execution_policy) {
  auto myExecutionSpace = Kokkos::DefaultExecutionSpace();
  auto myPolicyCreator = polk::ExecutionPolicyCreator().with(myExecutionSpace);

  static_assert(myPolicyCreator.getRank() == polk::unknownRank);
  static_assert(
      std::is_same_v<decltype(myPolicyCreator.getRange()), polk::UnknownRange>);
  static_assert(std::is_same_v<decltype(myPolicyCreator.getTiling()),
                               polk::UnknownTiling>);
  static_assert(std::is_same_v<decltype(myPolicyCreator.getExecutionSpace()),
                               Kokkos::DefaultExecutionSpace>);
}

TEST(test_execution_policy_creator, test_get_policy_mdrangepolicy) {
  auto myRange = polk::Range<2>({0, 0}, {1, 1});
  auto myPolicyCreator = polk::ExecutionPolicyCreator().with(myRange);
  auto policy = myPolicyCreator.getPolicy();

  static_assert(Kokkos::is_execution_policy<decltype(policy)>::value);
  static_assert(policy.rank == 2);

  ASSERT_EQ(policy.m_lower[0], 0);
  ASSERT_EQ(policy.m_lower[1], 0);
  ASSERT_EQ(policy.m_upper[0], 1);
  ASSERT_EQ(policy.m_upper[1], 1);
}

TEST(test_execution_policy_creator, test_get_policy_mdrangepolicy_tiling) {
  auto myRange = polk::Range<2>({0, 0}, {100, 100});
  auto myTiling = polk::Tiling<2>({10, 10});
  auto myPolicyCreator =
      polk::ExecutionPolicyCreator().with(myRange).with(myTiling);
  auto policy = myPolicyCreator.getPolicy();

  static_assert(Kokkos::is_execution_policy<decltype(policy)>::value);
  static_assert(policy.rank == 2);

  ASSERT_EQ(policy.m_lower[0], 0);
  ASSERT_EQ(policy.m_lower[1], 0);
  ASSERT_EQ(policy.m_upper[0], 100);
  ASSERT_EQ(policy.m_upper[1], 100);
  ASSERT_EQ(policy.m_tile[0], 10);
  ASSERT_EQ(policy.m_tile[1], 10);
}

TEST(test_execution_policy_creator, test_get_policy_mdrangepolicy_space) {
  auto myRange = polk::Range<2>({0, 0}, {1, 1});
  auto myExecutionSpace = Kokkos::DefaultExecutionSpace();
  auto myPolicyCreator =
      polk::ExecutionPolicyCreator().with(myRange).with(myExecutionSpace);
  auto policy = myPolicyCreator.getPolicy();

  static_assert(Kokkos::is_execution_policy<decltype(policy)>::value);
  static_assert(policy.rank == 2);
  static_assert(
      std::is_same_v<std::remove_const_t<
                         std::remove_reference_t<decltype(policy.space())>>,
                     std::remove_const_t<std::remove_reference_t<
                         Kokkos::DefaultExecutionSpace>>>);

  ASSERT_EQ(policy.m_lower[0], 0);
  ASSERT_EQ(policy.m_lower[1], 0);
  ASSERT_EQ(policy.m_upper[0], 1);
  ASSERT_EQ(policy.m_upper[1], 1);
}

TEST(test_execution_policy_creator,
     test_get_policy_mdrangepolicy_tiling_space) {
  auto myRange = polk::Range<2>({0, 0}, {100, 100});
  auto myTiling = polk::Tiling<2>({10, 10});
  auto myExecutionSpace = Kokkos::DefaultExecutionSpace();
  auto myPolicyCreator =
      polk::ExecutionPolicyCreator().with(myRange).with(myTiling).with(
          myExecutionSpace);
  auto policy = myPolicyCreator.getPolicy();

  static_assert(Kokkos::is_execution_policy<decltype(policy)>::value);
  static_assert(policy.rank == 2);
  static_assert(
      std::is_same_v<std::remove_const_t<
                         std::remove_reference_t<decltype(policy.space())>>,
                     std::remove_const_t<std::remove_reference_t<
                         Kokkos::DefaultExecutionSpace>>>);

  ASSERT_EQ(policy.m_lower[0], 0);
  ASSERT_EQ(policy.m_lower[1], 0);
  ASSERT_EQ(policy.m_upper[0], 100);
  ASSERT_EQ(policy.m_upper[1], 100);
  ASSERT_EQ(policy.m_tile[0], 10);
  ASSERT_EQ(policy.m_tile[1], 10);
}

TEST(test_execution_policy_creator, test_get_policy_rangepolicy) {
  auto myRange = polk::Range(0, 1);
  auto myPolicyCreator = polk::ExecutionPolicyCreator().with(myRange);
  auto policy = myPolicyCreator.getPolicy();

  static_assert(Kokkos::is_execution_policy<decltype(policy)>::value);

  ASSERT_EQ(policy.begin(), 0);
  ASSERT_EQ(policy.end(), 1);
}

TEST(test_execution_policy_creator, test_get_policy_rangepolicy_tiling) {
  auto myRange = polk::Range(0, 100);
  auto myTiling = polk::Tiling(10);
  auto myPolicyCreator =
      polk::ExecutionPolicyCreator().with(myRange).with(myTiling);
  auto policy = myPolicyCreator.getPolicy();

  static_assert(Kokkos::is_execution_policy<decltype(policy)>::value);

  ASSERT_EQ(policy.begin(), 0);
  ASSERT_EQ(policy.end(), 100);
  ASSERT_EQ(policy.chunk_size(), 10);
}

TEST(test_execution_policy_creator, test_get_policy_rangepolicy_space) {
  auto myRange = polk::Range(0, 1);
  auto myExecutionSpace = Kokkos::DefaultExecutionSpace();
  auto myPolicyCreator =
      polk::ExecutionPolicyCreator().with(myRange).with(myExecutionSpace);
  auto policy = myPolicyCreator.getPolicy();

  static_assert(Kokkos::is_execution_policy<decltype(policy)>::value);
  static_assert(
      std::is_same_v<std::remove_const_t<
                         std::remove_reference_t<decltype(policy.space())>>,
                     std::remove_const_t<std::remove_reference_t<
                         Kokkos::DefaultExecutionSpace>>>);

  ASSERT_EQ(policy.begin(), 0);
  ASSERT_EQ(policy.end(), 1);
}

TEST(test_execution_policy_creator, test_get_policy_rangepolicy_tiling_space) {
  auto myRange = polk::Range(0, 100);
  auto myTiling = polk::Tiling(10);
  auto myExecutionSpace = Kokkos::DefaultExecutionSpace();
  auto myPolicyCreator =
      polk::ExecutionPolicyCreator().with(myRange).with(myTiling).with(
          myExecutionSpace);
  auto policy = myPolicyCreator.getPolicy();

  static_assert(Kokkos::is_execution_policy<decltype(policy)>::value);
  static_assert(
      std::is_same_v<std::remove_const_t<
                         std::remove_reference_t<decltype(policy.space())>>,
                     std::remove_const_t<std::remove_reference_t<
                         Kokkos::DefaultExecutionSpace>>>);

  ASSERT_EQ(policy.begin(), 0);
  ASSERT_EQ(policy.end(), 100);
  ASSERT_EQ(policy.chunk_size(), 10);
}
