#ifndef NOCTA_LANG_LEXER_H
#define NOCTA_LANG_LEXER_H

#include "token.h"

typedef struct {
    const char* start;
    const char* current;
    int line;
} nc_lexer;

// Initialize lexer with source code
void nc_lexer_init(nc_lexer* lexer, const char* source);

// Scan next token
nc_token nc_lexer_scan_token(nc_lexer* lexer);

#endif // NOCTA_LANG_LEXER_H
