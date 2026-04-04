#pragma once

// ============================================================================
// Minimal test framework for heimdall.
//
// No external dependencies (no gtest, catch2, etc.).
// Just macros for TEST, ASSERT_TRUE, ASSERT_EQ, ASSERT_NEAR, ASSERT_THROWS.
// Tests self-register via static constructors.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace heimdall::test {

// ---------------------------------------------------------------------------
// Test registry
// ---------------------------------------------------------------------------

struct TestEntry {
    std::string name;
    std::function<void()> func;
    bool skip = false;
};

inline std::vector<TestEntry>& test_registry() {
    static std::vector<TestEntry> registry;
    return registry;
}

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<void()> func, bool skip = false) {
        test_registry().push_back({name, std::move(func), skip});
    }
};

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

inline int run_all_tests() {
    auto& tests = test_registry();
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    std::vector<std::string> failures;

    std::printf("\n=== heimdall test suite ===\n\n");

    for (auto& t : tests) {
        if (t.skip) {
            std::printf("  \033[33mSKIP\033[0m  %s\n", t.name.c_str());
            ++skipped;
            continue;
        }
        try {
            t.func();
            std::printf("  \033[32mPASS\033[0m  %s\n", t.name.c_str());
            ++passed;
        } catch (const std::exception& e) {
            std::printf("  \033[31mFAIL\033[0m  %s\n        %s\n",
                        t.name.c_str(), e.what());
            failures.push_back(t.name);
            ++failed;
        }
    }

    int total = passed + failed;
    std::printf("\n--- %d/%d tests passed", passed, total);
    if (skipped > 0) std::printf(", %d skipped", skipped);
    std::printf(" ---\n");

    if (!failures.empty()) {
        std::printf("\nFailed tests:\n");
        for (auto& f : failures) {
            std::printf("  - %s\n", f.c_str());
        }
    }
    std::printf("\n");

    return (failed > 0) ? 1 : 0;
}

} // namespace heimdall::test

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

#define TEST(name)                                                           \
    void test_##name();                                                      \
    static ::heimdall::test::TestRegistrar reg_##name(#name, test_##name);   \
    void test_##name()

#define TEST_SKIP(name)                                                          \
    void test_##name();                                                          \
    static ::heimdall::test::TestRegistrar reg_##name(#name, test_##name, true); \
    void test_##name()

#define ASSERT_TRUE(expr)                                                    \
    do {                                                                      \
        if (!(expr)) {                                                        \
            std::ostringstream oss_;                                           \
            oss_ << __FILE__ << ":" << __LINE__                               \
                 << ": ASSERT_TRUE failed: " #expr;                           \
            throw std::runtime_error(oss_.str());                             \
        }                                                                     \
    } while (0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b)                                                      \
    do {                                                                      \
        auto a_ = (a);                                                        \
        auto b_ = (b);                                                        \
        if (a_ != b_) {                                                       \
            std::ostringstream oss_;                                           \
            oss_ << __FILE__ << ":" << __LINE__                               \
                 << ": ASSERT_EQ failed: " #a " == " #b                       \
                 << " (" << a_ << " != " << b_ << ")";                        \
            throw std::runtime_error(oss_.str());                             \
        }                                                                     \
    } while (0)

#define ASSERT_NE(a, b)                                                      \
    do {                                                                      \
        auto a_ = (a);                                                        \
        auto b_ = (b);                                                        \
        if (a_ == b_) {                                                       \
            std::ostringstream oss_;                                           \
            oss_ << __FILE__ << ":" << __LINE__                               \
                 << ": ASSERT_NE failed: " #a " != " #b                       \
                 << " (both " << a_ << ")";                                    \
            throw std::runtime_error(oss_.str());                             \
        }                                                                     \
    } while (0)

#define ASSERT_NEAR(a, b, eps)                                               \
    do {                                                                      \
        auto a_ = (a);                                                        \
        auto b_ = (b);                                                        \
        if (std::abs(a_ - b_) > (eps)) {                                      \
            std::ostringstream oss_;                                           \
            oss_ << __FILE__ << ":" << __LINE__                               \
                 << ": ASSERT_NEAR failed: |" #a " - " #b "| <= " #eps        \
                 << " (|" << a_ << " - " << b_ << "| = "                      \
                 << std::abs(a_ - b_) << ")";                                  \
            throw std::runtime_error(oss_.str());                             \
        }                                                                     \
    } while (0)

#define ASSERT_GT(a, b)                                                      \
    do {                                                                      \
        auto a_ = (a);                                                        \
        auto b_ = (b);                                                        \
        if (!(a_ > b_)) {                                                     \
            std::ostringstream oss_;                                           \
            oss_ << __FILE__ << ":" << __LINE__                               \
                 << ": ASSERT_GT failed: " #a " > " #b                        \
                 << " (" << a_ << " <= " << b_ << ")";                        \
            throw std::runtime_error(oss_.str());                             \
        }                                                                     \
    } while (0)

#define ASSERT_GE(a, b)                                                      \
    do {                                                                      \
        auto a_ = (a);                                                        \
        auto b_ = (b);                                                        \
        if (!(a_ >= b_)) {                                                    \
            std::ostringstream oss_;                                           \
            oss_ << __FILE__ << ":" << __LINE__                               \
                 << ": ASSERT_GE failed: " #a " >= " #b                       \
                 << " (" << a_ << " < " << b_ << ")";                         \
            throw std::runtime_error(oss_.str());                             \
        }                                                                     \
    } while (0)

#define ASSERT_LE(a, b)                                                      \
    do {                                                                      \
        auto a_ = (a);                                                        \
        auto b_ = (b);                                                        \
        if (!(a_ <= b_)) {                                                    \
            std::ostringstream oss_;                                           \
            oss_ << __FILE__ << ":" << __LINE__                               \
                 << ": ASSERT_LE failed: " #a " <= " #b                        \
                 << " (" << a_ << " > " << b_ << ")";                         \
            throw std::runtime_error(oss_.str());                             \
        }                                                                     \
    } while (0)

#define ASSERT_THROWS(expr, exception_type)                                  \
    do {                                                                      \
        bool caught_ = false;                                                 \
        try { expr; } catch (const exception_type&) { caught_ = true; }       \
        if (!caught_) {                                                        \
            std::ostringstream oss_;                                           \
            oss_ << __FILE__ << ":" << __LINE__                               \
                 << ": ASSERT_THROWS failed: " #expr                           \
                 << " did not throw " #exception_type;                         \
            throw std::runtime_error(oss_.str());                             \
        }                                                                     \
    } while (0)
