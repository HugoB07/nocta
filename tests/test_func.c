#include "minunit.h"
#include "nocta/lang/compiler.h"
#include "nocta/lang/vm.h"
#include "nocta/lang/parser.h"
#include "nocta/lang/lexer.h"
#include "nocta/lang/chunk.h"
#include <string.h>

// Helper to run code and check result code
static nc_interpret_result compile_and_run(const char* source) {
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    
    nc_ast_program* prog = nc_parser_parse(&parser);
    if (!prog) return INTERPRET_COMPILE_ERROR;
    
    nc_chunk chunk;
    nc_chunk_init(&chunk);
    
    if (!nc_compile(prog, &chunk)) {
        nc_chunk_free(&chunk);
        nc_ast_free((nc_ast_node*)prog);
        return INTERPRET_COMPILE_ERROR;
    }
    
    nc_vm vm;
    nc_vm_init(&vm);
    
    nc_interpret_result result = nc_vm_interpret(&vm, &chunk);
    
    nc_vm_free(&vm);
    nc_chunk_free(&chunk);
    nc_ast_free((nc_ast_node*)prog);
    return result;
}

static char* test_func_basic() {
    const char* source = 
        "int add(int a, int b) { return a + b; }\n"
        "var result = add(10, 20);\n"
        "if (result != 30) { result = 0; } else { result = 1; }\n";
        
    mu_assert("Basic func call failed", compile_and_run(source) == INTERPRET_OK);
    return NULL;
}

static char* test_recursion() {
    const char* source = 
        "int fib(int n) {\n"
        "  if (n < 2) return n;\n"
        "  return fib(n - 1) + fib(n - 2);\n"
        "}\n"
        "var f = fib(10);\n";
        
    mu_assert("Recursion failed", compile_and_run(source) == INTERPRET_OK);
    return NULL;
}

char* test_func_suite() {
    mu_run_test(test_func_basic);
    mu_run_test(test_recursion);
    return NULL;
}
