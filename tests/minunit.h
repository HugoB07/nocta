#ifndef MINUNIT_H
#define MINUNIT_H

#include <stdio.h>
#include <time.h>

extern int tests_run;
extern int tests_failed;

#define mu_assert(message, test) do { if (!(test)) { \
    printf("FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
    tests_failed++; \
    return NULL; \
} } while (0)

#define mu_run_test(test) do { \
    printf("Running %s...\n", #test); \
    char *message = test(); \
    tests_run++; \
    if (message) { return message; } \
    else { printf("PASS: %s\n", #test); } \
} while (0)

#endif
