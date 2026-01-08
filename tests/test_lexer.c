#include "minunit.h"
#include "nocta/lang/lexer.h"
#include <string.h>

static char* test_lexer_symbols() {
    const char* source = "(){},.;=+-*/";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);

    nc_token t;
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect LPAREN", t.type == TOKEN_LPAREN);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect RPAREN", t.type == TOKEN_RPAREN);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect LBRACE", t.type == TOKEN_LBRACE);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect RBRACE", t.type == TOKEN_RBRACE);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect COMMA", t.type == TOKEN_COMMA);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect DOT", t.type == TOKEN_DOT);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect SEMICOLON", t.type == TOKEN_SEMICOLON);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect EQUAL", t.type == TOKEN_EQUAL);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect PLUS", t.type == TOKEN_PLUS);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect MINUS", t.type == TOKEN_MINUS);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect STAR", t.type == TOKEN_STAR);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect SLASH", t.type == TOKEN_SLASH);
    t = nc_lexer_scan_token(&lexer); mu_assert("Expect EOF", t.type == TOKEN_EOF);
    
    return NULL;
}

static char* test_lexer_keywords() {
    const char* source = "var int float tensor void if else return for while spawn";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    mu_assert("Expect VAR", nc_lexer_scan_token(&lexer).type == TOKEN_VAR);
    mu_assert("Expect INT", nc_lexer_scan_token(&lexer).type == TOKEN_INT);
    mu_assert("Expect FLOAT", nc_lexer_scan_token(&lexer).type == TOKEN_FLOAT);
    mu_assert("Expect TENSOR", nc_lexer_scan_token(&lexer).type == TOKEN_TENSOR);
    mu_assert("Expect VOID", nc_lexer_scan_token(&lexer).type == TOKEN_VOID);
    mu_assert("Expect IF", nc_lexer_scan_token(&lexer).type == TOKEN_IF);
    mu_assert("Expect ELSE", nc_lexer_scan_token(&lexer).type == TOKEN_ELSE);
    mu_assert("Expect RETURN", nc_lexer_scan_token(&lexer).type == TOKEN_RETURN);
    mu_assert("Expect FOR", nc_lexer_scan_token(&lexer).type == TOKEN_FOR);
    mu_assert("Expect WHILE", nc_lexer_scan_token(&lexer).type == TOKEN_WHILE);
    mu_assert("Expect SPAWN", nc_lexer_scan_token(&lexer).type == TOKEN_SPAWN);
    
    return NULL;
}

static char* test_lexer_identifiers() {
    const char* source = "batch_size learningRate _hidden";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    nc_token t = nc_lexer_scan_token(&lexer);
    mu_assert("Expect ID", t.type == TOKEN_IDENTIFIER);
    mu_assert("Expect batch_size", strncmp(t.start, "batch_size", t.length) == 0);
    
    t = nc_lexer_scan_token(&lexer);
    mu_assert("Expect ID", t.type == TOKEN_IDENTIFIER);
    
    t = nc_lexer_scan_token(&lexer);
    mu_assert("Expect ID", t.type == TOKEN_IDENTIFIER);
    
    return NULL;
}

static char* test_lexer_numbers() {
    const char* source = "123 3.14 0.001";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    nc_token t = nc_lexer_scan_token(&lexer);
    mu_assert("Expect NUMBER", t.type == TOKEN_NUMBER);
    mu_assert("Content 123", strncmp(t.start, "123", t.length) == 0);
    
    t = nc_lexer_scan_token(&lexer);
    mu_assert("Expect NUMBER", t.type == TOKEN_NUMBER);
    mu_assert("Content 3.14", strncmp(t.start, "3.14", t.length) == 0);
    
    t = nc_lexer_scan_token(&lexer);
    mu_assert("Expect NUMBER", t.type == TOKEN_NUMBER);
    mu_assert("Content 0.001", strncmp(t.start, "0.001", t.length) == 0);
    
    return NULL;
}

static char* test_lexer_full_statement() {
    // float lr = 0.01;
    const char* source = "float lr = 0.01;";
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    mu_assert("1", nc_lexer_scan_token(&lexer).type == TOKEN_FLOAT);
    mu_assert("2", nc_lexer_scan_token(&lexer).type == TOKEN_IDENTIFIER);
    mu_assert("3", nc_lexer_scan_token(&lexer).type == TOKEN_EQUAL);
    mu_assert("4", nc_lexer_scan_token(&lexer).type == TOKEN_NUMBER);
    mu_assert("5", nc_lexer_scan_token(&lexer).type == TOKEN_SEMICOLON);
    
    return NULL;
}

char* test_lexer_suite() {
    mu_run_test(test_lexer_symbols);
    mu_run_test(test_lexer_keywords);
    mu_run_test(test_lexer_identifiers);
    mu_run_test(test_lexer_numbers);
    mu_run_test(test_lexer_full_statement);
    return NULL;
}
