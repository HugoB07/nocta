#ifndef NOCTA_LANG_TOKEN_H
#define NOCTA_LANG_TOKEN_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Token types
typedef enum {
    // End of file
    TOKEN_EOF,
    // Error token
    TOKEN_ERROR,

    // Keywords
    TOKEN_VAR,      // var
    TOKEN_INT,      // int
    TOKEN_FLOAT,    // float
    TOKEN_STRING,   // string
    TOKEN_BOOL,     // bool
    TOKEN_TENSOR,   // tensor
    TOKEN_VOID,     // void
    TOKEN_FN,       // fn (reserved? or unused if we use C style only)
    TOKEN_RETURN,   // return
    TOKEN_IF,       // if
    TOKEN_ELSE,     // else
    TOKEN_FOR,      // for
    TOKEN_WHILE,    // while
    TOKEN_IN,       // in
    TOKEN_BREAK,    // break
    TOKEN_CONTINUE, // continue
    TOKEN_SPAWN,    // spawn
    TOKEN_TRUE,     // true
    TOKEN_FALSE,    // false
    TOKEN_NIL,      // nil
    TOKEN_CLASS,    // class
    TOKEN_THIS,     // this

    // Identifiers & Literals
    TOKEN_IDENTIFIER,   // xy, train, model
    TOKEN_NUMBER,       // 123, 1.23
    TOKEN_STRING_LIT,   // "hello"

    // Punctuation
    TOKEN_LPAREN,       // (
    TOKEN_RPAREN,       // )
    TOKEN_LBRACE,       // {
    TOKEN_RBRACE,       // }
    TOKEN_LBRACKET,     // [
    TOKEN_RBRACKET,     // ]
    TOKEN_COMMA,        // ,
    TOKEN_DOT,          // .
    TOKEN_SEMICOLON,    // ;
    TOKEN_COLON,        // :
    TOKEN_EQUAL,        // =
    
    // Operators
    TOKEN_PLUS,         // +
    TOKEN_MINUS,        // -
    TOKEN_STAR,         // *
    
    // Comparison
    TOKEN_SLASH,        // /
    TOKEN_PERCENT,      // %
    TOKEN_LT,           // <
    TOKEN_GT,           // >
    TOKEN_LE,           // <=
    TOKEN_GE,           // >=
    TOKEN_EQ,           // ==
    TOKEN_NE,           // !=
    TOKEN_AND,          // &&
    TOKEN_OR,           // ||
    TOKEN_NOT,          // !
    
    // Compound Assignment
    TOKEN_PLUS_EQUAL,   // +=
    TOKEN_MINUS_EQUAL,  // -=
    TOKEN_STAR_EQUAL,   // *=
    TOKEN_SLASH_EQUAL,  // /=
    TOKEN_PERCENT_EQUAL // %=

} nc_token_type;

typedef struct {
    nc_token_type type;
    const char* start; // Pointer to start of token in source
    size_t length;     // Length of token
    int line;          // Line number
} nc_token;

const char* nc_token_type_to_string(nc_token_type type);

#endif // NOCTA_LANG_TOKEN_H
