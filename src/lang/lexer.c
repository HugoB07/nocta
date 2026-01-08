#include "nocta/lang/lexer.h"
#include <string.h>
#include <ctype.h>

void nc_lexer_init(nc_lexer* lexer, const char* source) {
    lexer->start = source;
    lexer->current = source;
    lexer->line = 1;
}

static bool is_at_end(nc_lexer* lexer) {
    return *lexer->current == '\0';
}

static char advance(nc_lexer* lexer) {
    lexer->current++;
    return lexer->current[-1];
}

static char peek(nc_lexer* lexer) {
    return *lexer->current;
}

static char peek_next(nc_lexer* lexer) {
    if (is_at_end(lexer)) return '\0';
    return lexer->current[1];
}

static bool match(nc_lexer* lexer, char expected) {
    if (is_at_end(lexer)) return false;
    if (*lexer->current != expected) return false;
    lexer->current++;
    return true;
}

static nc_token make_token(nc_lexer* lexer, nc_token_type type) {
    nc_token token;
    token.type = type;
    token.start = lexer->start;
    token.length = (size_t)(lexer->current - lexer->start);
    token.line = lexer->line;
    return token;
}

static nc_token error_token(nc_lexer* lexer, const char* message) {
    nc_token token;
    token.type = TOKEN_ERROR;
    token.start = message;
    token.length = strlen(message);
    token.line = lexer->line;
    return token;
}

static void skip_whitespace(nc_lexer* lexer) {
    for (;;) {
        char c = peek(lexer);
        switch (c) {
            case ' ':
            case '\r':
            case '\t':
                advance(lexer);
                break;
            case '\n':
                lexer->line++;
                advance(lexer);
                break;
            case '/':
                if (peek_next(lexer) == '/') {
                    while (peek(lexer) != '\n' && !is_at_end(lexer)) advance(lexer);
                } else {
                    return;
                }
                break;
            default:
                return;
        }
    }
}

static nc_token_type check_keyword(nc_lexer* lexer, int start, int length, const char* rest, nc_token_type type) {
    if (lexer->current - lexer->start == start + length &&
        memcmp(lexer->start + start, rest, length) == 0) {
        return type;
    }
    return TOKEN_IDENTIFIER;
}

static nc_token_type identifier_type(nc_lexer* lexer) {
    switch (lexer->start[0]) {
        case 'a': return check_keyword(lexer, 1, 2, "nd", TOKEN_AND);
        case 'b': 
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'o': return check_keyword(lexer, 2, 2, "ol", TOKEN_BOOL);
                    case 'r': return check_keyword(lexer, 2, 3, "eak", TOKEN_BREAK);
                }
            }
            break;
        case 'c': 
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'l': return check_keyword(lexer, 2, 3, "ass", TOKEN_CLASS);
                    case 'o': return check_keyword(lexer, 2, 6, "ntinue", TOKEN_CONTINUE);
                }
            }
            break;
        case 'e': return check_keyword(lexer, 1, 3, "lse", TOKEN_ELSE);
        case 'f':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'a': return check_keyword(lexer, 2, 3, "lse", TOKEN_FALSE);
                    case 'o': return check_keyword(lexer, 2, 1, "r", TOKEN_FOR);
                    case 'l': return check_keyword(lexer, 2, 3, "oat", TOKEN_FLOAT);
                }
            }
            break;
        case 'i':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'f': return TOKEN_IF; // Specialized check not needed since it's 2 chars
                    case 'n':
                         if (lexer->current - lexer->start == 2) return TOKEN_IN; // 'in'
                         return check_keyword(lexer, 2, 1, "t", TOKEN_INT); // 'int'
                }
            }
            break;
        case 'n': return check_keyword(lexer, 1, 2, "il", TOKEN_NIL);
        case 'o': return check_keyword(lexer, 1, 1, "r", TOKEN_OR);
        case 'r': return check_keyword(lexer, 1, 5, "eturn", TOKEN_RETURN);
        case 's':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'p': return check_keyword(lexer, 2, 3, "awn", TOKEN_SPAWN);
                    case 't': return check_keyword(lexer, 2, 4, "ring", TOKEN_STRING);
                    // struct
                }
            }
            break;
        case 't':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'e': return check_keyword(lexer, 2, 4, "nsor", TOKEN_TENSOR);
                    case 'h': return check_keyword(lexer, 2, 2, "is", TOKEN_THIS);
                    case 'r': return check_keyword(lexer, 2, 2, "ue", TOKEN_TRUE);
                }
            }
            break;
        case 'v':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'a': return check_keyword(lexer, 2, 1, "r", TOKEN_VAR);
                    case 'o': return check_keyword(lexer, 2, 2, "id", TOKEN_VOID);
                }
            }
            break;
        case 'w': return check_keyword(lexer, 1, 4, "hile", TOKEN_WHILE);
    }
    return TOKEN_IDENTIFIER;
}

static nc_token identifier(nc_lexer* lexer) {
    while (isalnum(peek(lexer)) || peek(lexer) == '_') advance(lexer);
    return make_token(lexer, identifier_type(lexer));
}

static nc_token number(nc_lexer* lexer) {
    while (isdigit(peek(lexer))) advance(lexer);

    if (peek(lexer) == '.' && isdigit(peek_next(lexer))) {
        advance(lexer);
        while (isdigit(peek(lexer))) advance(lexer);
    }
    
    // Scientific notation
    if (peek(lexer) == 'e' || peek(lexer) == 'E') {
        advance(lexer); // Consume 'e'
        
        // Optional sign
        if (peek(lexer) == '+' || peek(lexer) == '-') {
            advance(lexer);
        }
        
        // Exponent digits
        if (!isdigit(peek(lexer))) {
            return error_token(lexer, "Unterminated scientific notation.");
        }
        while (isdigit(peek(lexer))) advance(lexer);
    }

    return make_token(lexer, TOKEN_NUMBER);
}

static nc_token string(nc_lexer* lexer) {
    while (peek(lexer) != '"' && !is_at_end(lexer)) {
        if (peek(lexer) == '\n') lexer->line++;
        advance(lexer);
    }

    if (is_at_end(lexer)) return error_token(lexer, "Unterminated string.");
    advance(lexer); // Closing quote
    return make_token(lexer, TOKEN_STRING_LIT);
}

nc_token nc_lexer_scan_token(nc_lexer* lexer) {
    skip_whitespace(lexer);
    lexer->start = lexer->current;

    if (is_at_end(lexer)) return make_token(lexer, TOKEN_EOF);

    char c = advance(lexer);
    
    if (isalpha(c) || c == '_') return identifier(lexer);
    if (isdigit(c)) return number(lexer);

    switch (c) {
        case '(': return make_token(lexer, TOKEN_LPAREN);
        case ')': return make_token(lexer, TOKEN_RPAREN);
        case '{': return make_token(lexer, TOKEN_LBRACE);
        case '}': return make_token(lexer, TOKEN_RBRACE);
        case '[': return make_token(lexer, TOKEN_LBRACKET);
        case ']': return make_token(lexer, TOKEN_RBRACKET);
        case ',': return make_token(lexer, TOKEN_COMMA);
        case '.': return make_token(lexer, TOKEN_DOT);
        case ';': return make_token(lexer, TOKEN_SEMICOLON);
        case ':': return make_token(lexer, TOKEN_COLON);
        
        case '+': return make_token(lexer, match(lexer, '=') ? TOKEN_PLUS_EQUAL : TOKEN_PLUS);
        case '-': return make_token(lexer, match(lexer, '=') ? TOKEN_MINUS_EQUAL : TOKEN_MINUS);
        case '*': return make_token(lexer, match(lexer, '=') ? TOKEN_STAR_EQUAL : TOKEN_STAR);
        case '/': return make_token(lexer, match(lexer, '=') ? TOKEN_SLASH_EQUAL : TOKEN_SLASH);
        case '%': return make_token(lexer, match(lexer, '=') ? TOKEN_PERCENT_EQUAL : TOKEN_PERCENT);
        
        case '!': return make_token(lexer, match(lexer, '=') ? TOKEN_NE : TOKEN_NOT);
        case '=': return make_token(lexer, match(lexer, '=') ? TOKEN_EQ : TOKEN_EQUAL);
        case '<': return make_token(lexer, match(lexer, '=') ? TOKEN_LE : TOKEN_LT);
        case '>': return make_token(lexer, match(lexer, '=') ? TOKEN_GE : TOKEN_GT);
        
        case '"': return string(lexer);
    }

    return error_token(lexer, "Unexpected character.");
}

const char* nc_token_type_to_string(nc_token_type type) {
    switch (type) {
        case TOKEN_VAR: return "TOKEN_VAR";
        case TOKEN_INT: return "TOKEN_INT";
        case TOKEN_FLOAT: return "TOKEN_FLOAT";
        case TOKEN_TENSOR: return "TOKEN_TENSOR";
        case TOKEN_VOID: return "TOKEN_VOID";
        case TOKEN_IDENTIFIER: return "TOKEN_IDENTIFIER";
        case TOKEN_NUMBER: return "TOKEN_NUMBER";
        case TOKEN_PLUS: return "TOKEN_PLUS";
        case TOKEN_EQUAL: return "TOKEN_EQUAL";
        case TOKEN_SEMICOLON: return "TOKEN_SEMICOLON";
        case TOKEN_CLASS: return "TOKEN_CLASS";
        case TOKEN_THIS: return "TOKEN_THIS";
        case TOKEN_EOF: return "TOKEN_EOF";
        default: return "TOKEN_OTHER";
    }
}
