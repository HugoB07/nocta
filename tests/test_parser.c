#include "minunit.h"
#include "nocta/lang/parser.h"
#include <string.h>

static char* test_parser_var_decl() {
    const char* source = "int x = 42;";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    
    nc_ast_program* prog = nc_parser_parse(&parser);
    
    mu_assert("Program NULL", prog != NULL);
    mu_assert("Stmt count mismatch", prog->n_stmts == 1);
    
    nc_ast_node* stmt = prog->statements[0];
    mu_assert("Stmt type mismatch", stmt->type == AST_VAR_DECL);
    
    nc_ast_var_decl* decl = (nc_ast_var_decl*)stmt;
    mu_assert("Var type mismatch", decl->type_tok.type == TOKEN_INT);
    mu_assert("Var name mismatch", strncmp(decl->name.start, "x", 1) == 0);
    mu_assert("Initializer NULL", decl->initializer != NULL);
    mu_assert("Initializer type mismatch", decl->initializer->type == AST_LITERAL);
    
    nc_ast_free((nc_ast_node*)prog);
    return NULL;
}

static char* test_parser_binary_expr() {
    // float y = 1 + 2 * 3;
    const char* source = "float y = 1 + 2 * 3;";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    
    nc_ast_program* prog = nc_parser_parse(&parser);
    
    mu_assert("Program NULL", prog != NULL);
    nc_ast_var_decl* decl = (nc_ast_var_decl*)prog->statements[0];
    
    nc_ast_binary* add = (nc_ast_binary*)decl->initializer;
    mu_assert("Root not binary", add->base.type == AST_BINARY);
    mu_assert("Root not +", add->operator.type == TOKEN_PLUS);
    
    nc_ast_literal* left = (nc_ast_literal*)add->left;
    mu_assert("Left not literal", left->base.type == AST_LITERAL);
    
    nc_ast_binary* mul = (nc_ast_binary*)add->right;
    mu_assert("Right not binary", mul->base.type == AST_BINARY);
    mu_assert("Right not *", mul->operator.type == TOKEN_STAR);
    
    nc_ast_free((nc_ast_node*)prog);
    return NULL;
}

static char* test_parser_assignment() {
    // x = 10;
    // Note: Parsing assignment as statement is not yet fully in top-level loop 
    // unless we treat it as expr_stmt.
    const char* source = "x = 10;";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    
    nc_ast_program* prog = nc_parser_parse(&parser);
    
    mu_assert("Program NULL", prog != NULL);
    nc_ast_expr_stmt* stmt = (nc_ast_expr_stmt*)prog->statements[0];
    mu_assert("Stmt not EXPR_STMT", stmt->base.type == AST_EXPR_STMT);
    
    nc_ast_assignment* assign = (nc_ast_assignment*)stmt->expr;
    mu_assert("Expr not assignment", assign->base.type == AST_ASSIGNMENT);
    mu_assert("Assign name wrong", strncmp(assign->name.start, "x", 1) == 0);
    
    nc_ast_free((nc_ast_node*)prog);
    return NULL;
}


char* test_parser_suite() {
    mu_run_test(test_parser_var_decl);
    mu_run_test(test_parser_binary_expr);
    mu_run_test(test_parser_assignment);
    return NULL;
}
