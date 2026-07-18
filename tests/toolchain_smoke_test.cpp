#include <gtest/gtest.h>

// Proves the CMake + GoogleTest wiring itself works (S0.2), independent of
// any MPPI code. Real coverage starts with S1.1's dynamics tests.
TEST(Toolchain, GTestLinksAndRuns) {
  ASSERT_EQ(1 + 1, 2);
}
