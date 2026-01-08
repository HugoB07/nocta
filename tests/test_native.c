#include "minunit.h"
#include "nocta/lang/vm.h"
#include "nocta/lang/compiler.h"
#include "nocta/lang/parser.h"
#include "nocta/core/tensor.h"
#include <string.h>

static bool called = false;
static double last_arg = 0;

static nc_value native_mock(nc_vm* vm, int arg_count, nc_value* args) {
    called = true;
    if (arg_count > 0 && IS_NUMBER(args[0])) {
        last_arg = AS_NUMBER(args[0]);
    }
    return NUMBER_VAL(123);
}

static char* test_native_call() {
    called = false;
    last_arg = 0;
    
    // Source: "mock(42);"
    const char* source = "mock(42);";
    
    // 1. Compile
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    nc_ast_program* prog = nc_parser_parse(&parser);
    
    if (!prog) return "Parse error";
    
    nc_chunk chunk;
    nc_chunk_init(&chunk);
    if (!nc_compile(prog, &chunk)) {
        return "Compile error";
    }
    
    // 2. Setup VM and Register Native
    nc_vm vm;
    nc_vm_init(&vm);
    nc_vm_define_native(&vm, "mock", native_mock);
    
    // 3. Interpret
    nc_interpret_result result = nc_vm_interpret(&vm, &chunk);
    mu_assert("Runtime error", result == INTERPRET_OK);
    
    // 4. Verify Side Effects
    mu_assert("Native not called", called);
    mu_assert("Arg not passed", last_arg == 42);
    
    // Cleanup
    nc_vm_free(&vm);
    nc_chunk_free(&chunk);
    nc_ast_free((nc_ast_node*)prog);
    return NULL;
}

// Helper native function to assert tensor properties from script
static nc_value native_assert_shape(nc_vm* vm, int arg_count, nc_value* args) {
    // args[0]: tensor
    // args[1]: expected ndim
    // args[2...]: expected shape dims
    
    if (!IS_TENSOR(args[0])) {
        fprintf(stderr, "Assertion failed: Argument 0 is not a tensor\n");
        exit(1); // Fail hard for test
    }
    nc_tensor* t = AS_TENSOR(args[0]);
    
    if (!IS_NUMBER(args[1])) { fprintf(stderr, "Arg 1 must be ndim\n"); exit(1); }
    int expected_ndim = (int)AS_NUMBER(args[1]);
    
    if (t->ndim != expected_ndim) {
        fprintf(stderr, "Assertion failed: Expected ndim %d, got %zu\n", expected_ndim, t->ndim);
        exit(1);
    }
    
    for (int i = 0; i < expected_ndim; i++) {
        if (!IS_NUMBER(args[2+i])) { fprintf(stderr, "Shape arg must be number\n"); exit(1); }
        int expected_dim = (int)AS_NUMBER(args[2+i]);
        if (t->shape[i] != expected_dim) {
             fprintf(stderr, "Assertion failed: Dim %d mismatch. Expected %d, got %zu\n", i, expected_dim, t->shape[i]);
             exit(1);
        }
    }
    return NIL_VAL;
}

static char* test_native_tensor_ops() {
    // WORKAROUND: For this test, valid until compiler is fixed:
    // We can nest calls if we don't have variables!
    // "assert_shape(matmul(randn(2,3), zeros(3,2)), 2, 2, 2);"
    
    const char* source_nested = 
        "assert_shape(randn(2, 3), 2, 2, 3); "
        "assert_shape(zeros(3, 2), 2, 3, 2); "
        "assert_shape(matmul(randn(2, 3), zeros(3, 2)), 2, 2, 2); ";

    nc_lexer lexer;
    nc_lexer_init(&lexer, source_nested);
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    nc_ast_program* prog = nc_parser_parse(&parser);
    mu_assert("Parse error", prog != NULL);
    
    nc_chunk chunk;
    nc_chunk_init(&chunk);
    bool compiled = nc_compile(prog, &chunk);
    mu_assert("Compile error", compiled);
    
    nc_vm vm;
    nc_vm_init(&vm); // Registers standard native functions
    nc_vm_define_native(&vm, "assert_shape", native_assert_shape);
    
    nc_interpret_result result = nc_vm_interpret(&vm, &chunk);
    mu_assert("Runtime error", result == INTERPRET_OK);
    
    nc_vm_free(&vm);
    nc_chunk_free(&chunk);
    nc_ast_free((nc_ast_node*)prog);
    return NULL;
}

char* test_native_suite() {
    mu_run_test(test_native_call);
    mu_run_test(test_native_tensor_ops);
    return NULL;
}
