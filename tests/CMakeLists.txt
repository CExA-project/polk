add_executable(
    test-polk
    main.cpp
    test.cpp
)

target_link_libraries(
    test-polk
    Polk::polk
    GTest::gtest
)

if(POLK_ENABLE_GTEST_DISCOVER_TESTS)
    gtest_discover_tests(test-polk)
endif()
