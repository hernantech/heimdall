// ============================================================================
// heimdall test runner entry point.
//
// All tests self-register via TestRegistrar in their respective .cpp files.
// This file just invokes the runner.
// ============================================================================

#include "test_helpers.h"

int main() {
    return heimdall::test::run_all_tests();
}
