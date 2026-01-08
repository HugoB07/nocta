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
    
    // For testing results, we ideally want to inspect VM globals or output.
    // We can define a native "capture" function or just return result code.
    // OR we relies on implicit return or side effects.
    // Let's use `visit` idiom from test_flow if possible, OR just rely on VM state inspection logic if we exposed it.
    // Since `compile_and_run` destroys VM, we can't inspect it.
    
    // Let's modify compile_and_run to define a hook?
    // Or just use the standard flow:
    nc_interpret_result result = nc_vm_interpret(&vm, &chunk);
    
    nc_vm_free(&vm);
    nc_chunk_free(&chunk);
    nc_ast_free((nc_ast_node*)prog);
    return result;
}

static char* test_for_loop_range() {
    // We rely on "if it doesn't crash, it might be working" for MVP unless we can assert values.
    // To assert values, we really need `test_flow` style visit/count.
    // But `compile_and_run` is self contained.
    
    // Let's copy `run_flow_test` logic basically but using standard includes.
    // Actually, `test_loop.c` can define its own runner with native capture.
    
    const char* source = 
        "var sum = 0; "
        "for (var i in range(0, 5)) { "
        "  sum = sum + i; "
        "} "
        "if (sum != 10) { 1/0; }"; // Trigger runtime error if fail
        
    mu_assert("For loop sum check failed", compile_and_run(source) == INTERPRET_OK);
    return NULL;
}

static char* test_empty_range() {
    const char* source = 
        "var count = 0; "
        "for (var i in range(0, 0)) { "
        "  count = count + 1; "
        "} "
        "if (count != 0) { 1/0; }";
        
    mu_assert("Empty range check failed", compile_and_run(source) == INTERPRET_OK);
    return NULL;
}

char* test_loop_suite() {
    mu_run_test(test_for_loop_range);
    mu_run_test(test_empty_range);
    return NULL;
}
