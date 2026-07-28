// A test framework small enough to read in one sitting.
//
// Deliberately dependency-free: the codec has no third-party dependencies, and
// the tests should not be the reason a build needs network access.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace gpudct_test {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

inline int& failures() {
  static int f = 0;
  return f;
}

inline std::string& current() {
  static std::string c;
  return c;
}

struct Register {
  Register(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline void fail(const char* file, int line, const std::string& msg) {
  ++failures();
  std::fprintf(stderr, "  FAIL %s:%d\n    %s\n", file, line, msg.c_str());
}

inline int run_all() {
  int failed_cases = 0;
  for (const Case& c : registry()) {
    current() = c.name;
    const int before = failures();
    std::printf("[ RUN  ] %s\n", c.name);
    c.fn();
    if (failures() != before) {
      ++failed_cases;
      std::printf("[ FAIL ] %s\n", c.name);
    } else {
      std::printf("[  OK  ] %s\n", c.name);
    }
  }
  std::printf("\n%zu tests, %d failed\n", registry().size(), failed_cases);
  return failed_cases == 0 ? 0 : 1;
}

}  // namespace gpudct_test

#define TEST(name)                                                       \
  static void name();                                                    \
  static ::gpudct_test::Register reg_##name(#name, &name);               \
  static void name()

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) ::gpudct_test::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_EQ(a, b)                                                   \
  do {                                                                   \
    const auto va_ = (a);                                                \
    const auto vb_ = (b);                                                \
    if (!(va_ == vb_))                                                   \
      ::gpudct_test::fail(__FILE__, __LINE__,                            \
                          std::string(#a " == " #b "  (") + std::to_string(va_) + \
                              " vs " + std::to_string(vb_) + ")");       \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                            \
  do {                                                                   \
    const double va_ = static_cast<double>(a);                           \
    const double vb_ = static_cast<double>(b);                           \
    if (!(std::fabs(va_ - vb_) <= static_cast<double>(tol)))             \
      ::gpudct_test::fail(__FILE__, __LINE__,                            \
                          std::string(#a " ~= " #b "  (") + std::to_string(va_) + \
                              " vs " + std::to_string(vb_) + ", tol " +  \
                              std::to_string(static_cast<double>(tol)) + ")"); \
  } while (0)

#define REQUIRE(cond)                                                    \
  do {                                                                   \
    if (!(cond)) {                                                       \
      ::gpudct_test::fail(__FILE__, __LINE__, "REQUIRE(" #cond ")");     \
      return;                                                            \
    }                                                                    \
  } while (0)

#define TEST_MAIN() \
  int main() { return ::gpudct_test::run_all(); }
