#include "minunit.h"
#include "nocta/lang/compiler.h"
#include "nocta/lang/parser.h"

static char* test_compile_arithmetic() {
    const char* source = "1 + 2 * 3;";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    nc_ast_program* prog = nc_parser_parse(&parser);
    
    nc_chunk chunk;
    nc_chunk_init(&chunk);
    
    bool result = nc_compile(prog, &chunk);
    mu_assert("Compilation failed", result);
    
    // Expected:
    // CONST 1
    // CONST 2
    // CONST 3
    // MUL
    // ADD
    // POP (Expression stmt)
    // RETURN (Implicit)
    
    // Note: We need to check opcode sequence
    int i = 0;
    mu_assert("Op 0 not CONST", chunk.code[i++] == OP_CONSTANT); i++; // Skip index
    mu_assert("Op 2 not CONST", chunk.code[i++] == OP_CONSTANT); i++;
    mu_assert("Op 4 not CONST", chunk.code[i++] == OP_CONSTANT); i++;
    mu_assert("Op 6 not MUL",   chunk.code[i++] == OP_MUL);
    mu_assert("Op 7 not ADD",   chunk.code[i++] == OP_ADD);
    mu_assert("Op 8 not POP",   chunk.code[i++] == OP_POP);
    mu_assert("Op 9 not NIL",   chunk.code[i++] == OP_NIL);
    mu_assert("Op 10 not RETURN", chunk.code[i++] == OP_RETURN);
    
    printf("\n");
    nc_chunk_disassemble(&chunk, "test_compile_arithmetic");

    nc_chunk_free(&chunk);
    nc_ast_free((nc_ast_node*)prog);
    return NULL;
}

char* test_compiler_suite() {
    mu_run_test(test_compile_arithmetic);
    return NULL;
}
