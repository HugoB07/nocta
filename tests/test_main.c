#include <stdio.h>
#include <string.h>
#include <time.h>
#include "minunit.h"
#include "nocta/nocta.h"

// Globals definitions
int tests_run = 0;
int tests_failed = 0;

// Forward declare test suites
extern char* test_tensor_suite();
extern char* test_ops_suite();
extern char* test_autograd_suite();
extern char* test_nn_suite();
extern char* test_optim_suite();
extern char* test_io_suite();
extern char* test_reduction_suite();
extern char* test_lexer_suite();
extern char* test_parser_suite();
extern char* test_chunk_suite();
extern char* test_compiler_suite();
extern char* test_vm_suite();
extern char* test_native_suite();
extern char* test_flow_suite();
extern char* test_func_suite();
extern char* test_loop_suite();

void run_suite(const char* name, char* (*suite_func)()) {
    printf("\n--- %s Tests ---\n", name);
    char* result = suite_func();
    if (result) {
        printf("%s suite failed\n", name);
    }
}

int main(int argc, char **argv) {
    nc_init();
    
    printf("=== Running Tests ===\n");
    clock_t start = clock();
    
    if (argc < 2 || strcmp(argv[1], "all") == 0) {
        run_suite("Tensor", test_tensor_suite);
        run_suite("Ops", test_ops_suite);
        run_suite("Autograd", test_autograd_suite);
        run_suite("NN", test_nn_suite);
        run_suite("Optim", test_optim_suite);
        run_suite("IO", test_io_suite);
        run_suite("Reduction", test_reduction_suite);
        run_suite("Lexer", test_lexer_suite);
        run_suite("Parser", test_parser_suite);
        run_suite("Chunk", test_chunk_suite);
        run_suite("Compiler", test_compiler_suite);
        run_suite("VM", test_vm_suite);
        run_suite("Native", test_native_suite);
        run_suite("Flow", test_flow_suite);
        run_suite("Func", test_func_suite);
        run_suite("Loop", test_loop_suite);
    } 
    else if (strcmp(argv[1], "tensor") == 0) {
        run_suite("Tensor", test_tensor_suite);
    }
    else if (strcmp(argv[1], "ops") == 0) {
        run_suite("Ops", test_ops_suite);
    }
    else if (strcmp(argv[1], "autograd") == 0) {
        run_suite("Autograd", test_autograd_suite);
    }
    else if (strcmp(argv[1], "nn") == 0) {
        run_suite("NN", test_nn_suite);
    }
    else if (strcmp(argv[1], "optim") == 0) {
        run_suite("Optim", test_optim_suite);
    }
    else if (strcmp(argv[1], "io") == 0) {
        run_suite("IO", test_io_suite);
    }
    else if (strcmp(argv[1], "reduction") == 0) {
        run_suite("Reduction", test_reduction_suite);
    }
    else if (strcmp(argv[1], "lexer") == 0) {
        run_suite("Lexer", test_lexer_suite);
    }
    else if (strcmp(argv[1], "parser") == 0) {
        run_suite("Parser", test_parser_suite);
    }
    else if (strcmp(argv[1], "chunk") == 0) {
        run_suite("Chunk", test_chunk_suite);
    }
    else if (strcmp(argv[1], "compiler") == 0) {
        run_suite("Compiler", test_compiler_suite);
    }
    else if (strcmp(argv[1], "vm") == 0) {
        run_suite("VM", test_vm_suite);
    }
    else if (strcmp(argv[1], "native") == 0) {
        run_suite("Native", test_native_suite);
    }
    else if (strcmp(argv[1], "flow") == 0) {
        run_suite("Flow", test_flow_suite);
    }
    else if (strcmp(argv[1], "func") == 0) {
        run_suite("Func", test_func_suite);
    }
    else if (strcmp(argv[1], "loop") == 0) {
        run_suite("Loop", test_loop_suite);
    }
    else {
        printf("Unknown suite: %s\n", argv[1]);
        nc_cleanup();
        return 1;
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests failed: %d\n", tests_failed);
    printf("Time taken: %.4f seconds\n", time_taken);
    
    nc_cleanup();
    return tests_failed != 0;
}
