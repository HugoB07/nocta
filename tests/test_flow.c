#include "minunit.h"
#include "nocta/lang/vm.h"
#include "nocta/lang/compiler.h"
#include "nocta/lang/parser.h"

static int visit_count = 0;
static int last_visited_id = 0;

static nc_value native_visit(nc_vm* vm, int arg_count, nc_value* args) {
    visit_count++;
    if (arg_count > 0 && IS_NUMBER(args[0])) {
        last_visited_id = (int)AS_NUMBER(args[0]);
    }
    return NIL_VAL;
}

static char* run_flow_test(const char* source) {
    visit_count = 0;
    last_visited_id = 0;
    
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    nc_ast_program* prog = nc_parser_parse(&parser);
    mu_assert("Parse error", prog != NULL);
    
    nc_chunk chunk;
    nc_chunk_init(&chunk);
    bool compiled = nc_compile(prog, &chunk);
    mu_assert("Compile error", compiled);
    
    nc_vm vm;
    nc_vm_init(&vm);
    nc_vm_define_native(&vm, "visit", native_visit);
    
    nc_interpret_result result = nc_vm_interpret(&vm, &chunk);
    mu_assert("Runtime error", result == INTERPRET_OK);
    
    nc_vm_free(&vm);
    nc_chunk_free(&chunk);
    nc_ast_free((nc_ast_node*)prog);
    return NULL;
}

static char* test_if_true() {
    const char* source = "if (true) { visit(1); } else { visit(2); }";
    char* err = run_flow_test(source);
    if (err) return err;
    
    mu_assert("Should visit then branch", visit_count == 1);
    mu_assert("Should visit ID 1", last_visited_id == 1);
    return NULL;
}

static char* test_if_false() {
    const char* source = "if (false) { visit(1); } else { visit(2); }";
    char* err = run_flow_test(source);
    if (err) return err;
    
    mu_assert("Should visit else branch", visit_count == 1);
    mu_assert("Should visit ID 2", last_visited_id == 2);
    return NULL;
}

static char* test_while_loop() {
    // "var i = 0; while (i < 3) { visit(i); i = i + 1; }"
    const char* source = 
        "var i = 0; "
        "while (i < 3) { "
        "  visit(i); "
        "  i = i + 1; "
        "}";
        
    char* err = run_flow_test(source);
    if (err) return err;
    
    mu_assert("Should have visited 3 times", visit_count == 3);
    mu_assert("Last visited ID should be 2", last_visited_id == 2);
    return NULL;
}

char* test_flow_suite() {
    mu_run_test(test_if_true);
    mu_run_test(test_if_false);
    mu_run_test(test_while_loop);
    return NULL;
}
