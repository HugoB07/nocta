#ifndef NOCTA_LANG_PARSER_H
#define NOCTA_LANG_PARSER_H

#include "lexer.h"
#include "ast.h"

typedef struct {
    nc_lexer* lexer;
    nc_token previous;
    nc_token current;
    bool had_error;
    bool panic_mode;
} nc_parser;

// Initialize parser
void nc_parser_init(nc_parser* parser, nc_lexer* lexer);

// Parse entry point
nc_ast_program* nc_parser_parse(nc_parser* parser);

#endif // NOCTA_LANG_PARSER_H
