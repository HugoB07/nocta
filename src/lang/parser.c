#include "nocta/lang/parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// Error Handling & Helpers
// ============================================================

static void error_at(nc_parser* parser, nc_token* token, const char* message) {
    if (parser->panic_mode) return;
    parser->panic_mode = true;
    
    fprintf(stderr, "[line %d] Error", token->line);

    if (token->type == TOKEN_EOF) {
        fprintf(stderr, " at end");
    } else if (token->type == TOKEN_ERROR) {
        // Nothing.
    } else {
        fprintf(stderr, " at '%.*s'", (int)token->length, token->start);
    }

    fprintf(stderr, ": %s\n", message);
    parser->had_error = true;
}

static void advance(nc_parser* parser) {
    parser->previous = parser->current;

    for (;;) {
        parser->current = nc_lexer_scan_token(parser->lexer);
        if (parser->current.type != TOKEN_ERROR) break;

        error_at(parser, &parser->current, parser->current.start);
    }
}

static void consume(nc_parser* parser, nc_token_type type, const char* message) {
    if (parser->current.type == type) {
        advance(parser);
        return;
    }

    error_at(parser, &parser->current, message);
}

static bool match_token(nc_parser* parser, nc_token_type type) {
    if (parser->current.type != type) return false;
    advance(parser);
    return true;
}

static bool check_token(nc_parser* parser, nc_token_type type) {
    return parser->current.type == type;
}

void nc_parser_init(nc_parser* parser, nc_lexer* lexer) {
    parser->lexer = lexer;
    parser->had_error = false;
    parser->panic_mode = false;
    advance(parser); // Prime the first token
}

// ============================================================
// Allocation Helpers
// ============================================================

static void* ast_alloc(size_t size) {
    void* p = malloc(size);
    if (!p) {
        fprintf(stderr, "Parser out of memory\n");
        exit(1);
    }
    memset(p, 0, size);
    return p;
}

#define ALLOC_NODE(T, type_enum) \
    T* node = (T*)ast_alloc(sizeof(T)); \
    node->base.type = type_enum;

// ============================================================
// Freeing
// ============================================================

void nc_ast_free(nc_ast_node* node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_LITERAL:
        case AST_VAR_ACCESS:
            free(node);
            break;
            
        case AST_BINARY: {
            nc_ast_binary* binary = (nc_ast_binary*)node;
            nc_ast_free(binary->left);
            nc_ast_free(binary->right);
            free(node);
            break;
        }
        case AST_UNARY: {
            nc_ast_unary* unary = (nc_ast_unary*)node;
            nc_ast_free(unary->right);
            free(node);
            break;
        }
        case AST_ASSIGNMENT: {
            nc_ast_assignment* assign = (nc_ast_assignment*)node;
            nc_ast_free(assign->value);
            free(node);
            break;
        }
        case AST_EXPR_STMT: {
            nc_ast_expr_stmt* stmt = (nc_ast_expr_stmt*)node;
            nc_ast_free(stmt->expr);
            free(node);
            break;
        }
        case AST_VAR_DECL: {
            nc_ast_var_decl* decl = (nc_ast_var_decl*)node;
            if (decl->initializer) nc_ast_free(decl->initializer);
            free(node);
            break;
        }
        case AST_BLOCK: {
            nc_ast_block* block = (nc_ast_block*)node;
            for (size_t i = 0; i < block->n_stmts; i++) {
                nc_ast_free(block->statements[i]);
            }
            free(block->statements);
            free(node);
            break;
        }
        case AST_IF: {
            nc_ast_if* stmt = (nc_ast_if*)node;
            nc_ast_free(stmt->condition);
            nc_ast_free(stmt->then_branch);
            if (stmt->else_branch) nc_ast_free(stmt->else_branch);
            free(node);
            break;
        }
        case AST_WHILE: {
            nc_ast_while* stmt = (nc_ast_while*)node;
            nc_ast_free(stmt->condition);
            nc_ast_free(stmt->body);
            free(node);
            break;
        }
        case AST_FUNCTION: {
            nc_ast_function* func = (nc_ast_function*)node;
            // Name is a token, no free needed
            // Params: array of tokens, free array
            if (func->params) free(func->params);
            nc_ast_free(func->body);
            free(node);
            break;
        }
        case AST_RETURN: {
            nc_ast_return* ret = (nc_ast_return*)node;
            if (ret->value) nc_ast_free(ret->value);
            free(node);
            break;
        }
        case AST_LIST_LITERAL: {
            nc_ast_list_literal* list = (nc_ast_list_literal*)node;
            for (size_t i = 0; i < list->element_count; i++) {
                nc_ast_free(list->elements[i]);
            }
            if (list->elements) free(list->elements);
            free(node);
            break;
        }
        case AST_INDEX_ACCESS: {
            nc_ast_index_access* access = (nc_ast_index_access*)node;
            nc_ast_free(access->object);
            nc_ast_free(access->index);
            free(node);
            break;
        }
        case AST_INDEX_SET: {
            nc_ast_index_set* set = (nc_ast_index_set*)node;
            nc_ast_free(set->object);
            nc_ast_free(set->index);
            nc_ast_free(set->value);
            free(node);
            break;
        }
        case AST_FOR_IN: {
            nc_ast_for_in* loop = (nc_ast_for_in*)node;
            nc_ast_free(loop->iterable);
            nc_ast_free(loop->body);
            free(node);
            break;
        }
        default:
            free(node);
            break;
    }
}

// ============================================================
// Expressions
// ============================================================

// Forward declarations
static nc_ast_node* expression(nc_parser* parser);
static nc_ast_node* assignment(nc_parser* parser);
static nc_ast_node* declaration(nc_parser* parser);

static nc_ast_node* primary(nc_parser* parser) {
    if (match_token(parser, TOKEN_FALSE) || match_token(parser, TOKEN_TRUE) ||
        match_token(parser, TOKEN_NUMBER) || match_token(parser, TOKEN_STRING_LIT) ||
        match_token(parser, TOKEN_NIL)) {
        ALLOC_NODE(nc_ast_literal, AST_LITERAL);
        node->token = parser->previous;
        return (nc_ast_node*)node;
    }

    if (match_token(parser, TOKEN_IDENTIFIER)) {
        ALLOC_NODE(nc_ast_var_access, AST_VAR_ACCESS);
        node->name = parser->previous;
        return (nc_ast_node*)node;
    }

    if (match_token(parser, TOKEN_LBRACKET)) {
        // List Literal: [ expr, expr ]
        ALLOC_NODE(nc_ast_list_literal, AST_LIST_LITERAL);
        
        node->elements = NULL;
        node->element_count = 0;
        
        if (!check_token(parser, TOKEN_RBRACKET)) {
            size_t capacity = 0;
            do {
                if (capacity < node->element_count + 1) {
                    capacity = (capacity < 8) ? 8 : capacity * 2;
                    node->elements = realloc(node->elements, sizeof(nc_ast_node*) * capacity);
                }
                node->elements[node->element_count++] = expression(parser);
            } while (match_token(parser, TOKEN_COMMA));
        }
        
        consume(parser, TOKEN_RBRACKET, "Expect ']' after list elements.");
        return (nc_ast_node*)node;
    }

    if (match_token(parser, TOKEN_THIS)) {
        ALLOC_NODE(nc_ast_var_access, AST_VAR_ACCESS);
        node->name = parser->previous;
        return (nc_ast_node*)node;
    }

    if (match_token(parser, TOKEN_LPAREN)) {
        nc_ast_node* expr = expression(parser);
        consume(parser, TOKEN_RPAREN, "Expect ')' after expression.");
        return expr;
    }

    error_at(parser, &parser->current, "Expect expression.");
    return NULL;
}

static nc_ast_node* call(nc_parser* parser) {
    nc_ast_node* expr = primary(parser);

    for (;;) {
        if (match_token(parser, TOKEN_LPAREN)) {
            // Function call
            ALLOC_NODE(nc_ast_call, AST_CALL);
            node->callee = expr;
            
            node->n_args = 0;
            node->args = NULL;
            
            if (parser->current.type != TOKEN_RPAREN) {
                // Use a temporary fixed-size buffer for parsing or dynamic
                // For MVP, simple realloc or max args
                #define MAX_ARGS 255
                nc_ast_node* args[MAX_ARGS];
                
                do {
                    if (node->n_args >= MAX_ARGS) {
                        error_at(parser, &parser->current, "Cannot have more than 255 arguments.");
                    }
                    nc_ast_node* arg = expression(parser);
                    if (node->n_args < MAX_ARGS) {
                        args[node->n_args++] = arg;
                    }
                } while (match_token(parser, TOKEN_COMMA));
                
                // Copy to node
                node->args = (nc_ast_node**)malloc(sizeof(nc_ast_node*) * node->n_args);
                memcpy(node->args, args, sizeof(nc_ast_node*) * node->n_args);
            }
            
            consume(parser, TOKEN_RPAREN, "Expect ')' after arguments.");
            expr = (nc_ast_node*)node;
        } else if (match_token(parser, TOKEN_DOT)) {
            // Member access or property get
            consume(parser, TOKEN_IDENTIFIER, "Expect property name after '.'.");
            nc_token name = parser->previous;
            
            // This is primarily Get Member.
            // But if it's on LHS of assignment, it becomes Set Member.
            // We return AST_MEMBER_ACCESS here. 
            // The assignment parser must handle conversion.
            ALLOC_NODE(nc_ast_member_access, AST_MEMBER_ACCESS);
            node->object = expr;
            node->member = name;
            expr = (nc_ast_node*)node;
        } else if (match_token(parser, TOKEN_LBRACKET)) {
            // Index Access: expr[index]
            nc_ast_node* index = expression(parser);
            consume(parser, TOKEN_RBRACKET, "Expect ']' after index.");
            
            ALLOC_NODE(nc_ast_index_access, AST_INDEX_ACCESS);
            node->object = expr;
            node->index = index;
            expr = (nc_ast_node*)node;
        } else {
            break;
        }
    }
    return expr;
}

static nc_ast_node* unary(nc_parser* parser) {
    if (match_token(parser, TOKEN_NOT) || match_token(parser, TOKEN_MINUS)) {
        nc_token operator = parser->previous;
        nc_ast_node* right = unary(parser);
        ALLOC_NODE(nc_ast_unary, AST_UNARY);
        node->operator = operator;
        node->right = right;
        return (nc_ast_node*)node;
    }
    return call(parser);
}

static nc_ast_node* multiplication(nc_parser* parser) {
    nc_ast_node* expr = unary(parser);

    while (match_token(parser, TOKEN_SLASH) || match_token(parser, TOKEN_STAR) || 
           match_token(parser, TOKEN_PERCENT)) {
        nc_token operator = parser->previous;
        nc_ast_node* right = unary(parser);
        ALLOC_NODE(nc_ast_binary, AST_BINARY);
        node->operator = operator;
        node->left = expr;
        node->right = right;
        expr = (nc_ast_node*)node;
    }
    return expr;
}

static nc_ast_node* addition(nc_parser* parser) {
    nc_ast_node* expr = multiplication(parser);

    while (match_token(parser, TOKEN_MINUS) || match_token(parser, TOKEN_PLUS)) {
        nc_token operator = parser->previous;
        nc_ast_node* right = multiplication(parser);
        ALLOC_NODE(nc_ast_binary, AST_BINARY);
        node->operator = operator;
        node->left = expr;
        node->right = right;
        expr = (nc_ast_node*)node;
    }
    return expr;
}

static nc_ast_node* comparison(nc_parser* parser) {
    nc_ast_node* expr = addition(parser);

    while (match_token(parser, TOKEN_GT) || match_token(parser, TOKEN_GE) ||
           match_token(parser, TOKEN_LT) || match_token(parser, TOKEN_LE)) {
        nc_token operator = parser->previous;
        nc_ast_node* right = addition(parser);
        ALLOC_NODE(nc_ast_binary, AST_BINARY);
        node->operator = operator;
        node->left = expr;
        node->right = right;
        expr = (nc_ast_node*)node;
    }
    return expr;
}

static nc_ast_node* equality(nc_parser* parser) {
    nc_ast_node* expr = comparison(parser);

    while (match_token(parser, TOKEN_NE) || match_token(parser, TOKEN_EQ)) {
        nc_token operator = parser->previous;
        nc_ast_node* right = comparison(parser);
        ALLOC_NODE(nc_ast_binary, AST_BINARY);
        node->operator = operator;
        node->left = expr;
        node->right = right;
        expr = (nc_ast_node*)node;
    }
    return expr;
}

static nc_ast_node* logic_and(nc_parser* parser) {
    nc_ast_node* expr = equality(parser);

    while (match_token(parser, TOKEN_AND)) {
        nc_token operator = parser->previous;
        nc_ast_node* right = equality(parser);
        ALLOC_NODE(nc_ast_logical, AST_LOGICAL);
        node->operator = operator;
        node->left = expr;
        node->right = right;
        expr = (nc_ast_node*)node;
    }
    return expr;
}

static nc_ast_node* logic_or(nc_parser* parser) {
    nc_ast_node* expr = logic_and(parser);

    while (match_token(parser, TOKEN_OR)) {
        nc_token operator = parser->previous;
        nc_ast_node* right = logic_and(parser);
        ALLOC_NODE(nc_ast_logical, AST_LOGICAL);
        node->operator = operator;
        node->left = expr;
        node->right = right;
        expr = (nc_ast_node*)node;
    }
    return expr;
}

static nc_ast_node* assignment(nc_parser* parser) {
    nc_ast_node* expr = logic_or(parser);

    if (match_token(parser, TOKEN_EQUAL) ||
        match_token(parser, TOKEN_PLUS_EQUAL) ||
        match_token(parser, TOKEN_MINUS_EQUAL) ||
        match_token(parser, TOKEN_STAR_EQUAL) ||
        match_token(parser, TOKEN_SLASH_EQUAL)) {
        
        nc_token equals = parser->previous; // The operator token
        nc_ast_node* value = assignment(parser);

        if (expr->type == AST_VAR_ACCESS) {
            nc_token name = ((nc_ast_var_access*)expr)->name;
            
            // ... (Compound assignment check is fine with VarAccess too) ...
            if (equals.type != TOKEN_EQUAL) {
                 // Reuse binary op logic from previous step
                 ALLOC_NODE(nc_ast_binary, AST_BINARY);
                 node->left = expr;
                 node->right = value;
                 switch (equals.type) {
                    case TOKEN_PLUS_EQUAL: node->operator.type = TOKEN_PLUS; break;
                    case TOKEN_MINUS_EQUAL: node->operator.type = TOKEN_MINUS; break;
                    case TOKEN_STAR_EQUAL: node->operator.type = TOKEN_STAR; break;
                    case TOKEN_SLASH_EQUAL: node->operator.type = TOKEN_SLASH; break;
                    default: break; 
                 }
                 value = (nc_ast_node*)node;
            } else {
                free(expr);
            }

            // Create Assignment Node
            ALLOC_NODE(nc_ast_assignment, AST_ASSIGNMENT);
            node->name = name;
            node->value = value;
            
            return (nc_ast_node*)node;
        } else if (expr->type == AST_MEMBER_ACCESS) {
            nc_ast_member_access* get = (nc_ast_member_access*)expr;
            
            /*
            if (equals.type != TOKEN_EQUAL) {
                // Compound assignment on property
                // obj.x += 1
                // Desugar to: obj.x = obj.x + 1
                // We need to duplicate `obj.x` access or ensure standard evaluation order logic.
                // AST_SET_MEMBER { object: expr->object, member: expr->member, value: BINARY(GET(obj,member), op, value) }
                
                // Issue: If `obj` is an expression with side effects (e.g. `getObj().x += 1`),
                // evaluating `obj` twice is bad. 
                // Creating a proper AST for compound setting is better, or backend handles it.
                // For MVP: We assume simple object expressions or accept side effect double eval if simple desugaring.
                // Or we can fail compound assignment on members for now or just implement it recklessly reusing the pointer.
                // Reusing AST node pointer `get->object` in two places (Left of Binary AND in SetMember) 
                // might cause double free if we aren't careful.
                
                // Let's implement STANDARD assignment `obj.x = y` first properly.
                // Compound on properties is advanced.
                // Let's ERROR on compound property assignment for now to be safe.
                error_at(parser, &equals, "Compound assignment not supported on properties yet.");
                return expr;
            }
            */

            ALLOC_NODE(nc_ast_set_member, AST_SET_MEMBER);
            node->object = get->object;
            node->member = get->member;
            node->value = value;
            node->operator = equals; // Store the operator (EQUAL or PLUS_EQUAL etc)
            
            free(get); // Free the Get Member shell
            return (nc_ast_node*)node;
        }
        else if (expr->type == AST_INDEX_ACCESS) {
            nc_ast_index_access* get = (nc_ast_index_access*)expr;
            
            ALLOC_NODE(nc_ast_index_set, AST_INDEX_SET);
            node->object = get->object;
            node->index = get->index;
            node->value = value;
            
            free(get); // free shell
            return (nc_ast_node*)node;
        }
        
        error_at(parser, &equals, "Invalid assignment target.");
    }
    return expr;
}


static nc_ast_node* expression(nc_parser* parser) {
    return assignment(parser);
}

// ============================================================
// Statements
// ============================================================

static nc_ast_node* var_declaration(nc_parser* parser) {
    // We arrive here after consuming type (int, float, var, etc.)
    nc_token type = parser->previous;
    
    // Support generic tensor<...>
    if (type.type == TOKEN_TENSOR && match_token(parser, TOKEN_LT)) {
        // tensor<...>
        // consume dimensions...
        // For draft, consume until GT
        while (!match_token(parser, TOKEN_GT) && !match_token(parser, TOKEN_EOF)) advance(parser);
    }
    
    consume(parser, TOKEN_IDENTIFIER, "Expect variable name.");
    nc_token name = parser->previous;
    
    nc_ast_node* initializer = NULL;
    if (match_token(parser, TOKEN_EQUAL)) {
        initializer = expression(parser);
    }
    
    consume(parser, TOKEN_SEMICOLON, "Expect ';' after variable declaration.");
    
    ALLOC_NODE(nc_ast_var_decl, AST_VAR_DECL);
    node->type_tok = type;
    node->name = name;
    node->initializer = initializer;
    return (nc_ast_node*)node;
}

static nc_ast_node* expression_statement(nc_parser* parser) {
    nc_ast_node* expr = expression(parser);
    consume(parser, TOKEN_SEMICOLON, "Expect ';' after expression.");
    
    ALLOC_NODE(nc_ast_expr_stmt, AST_EXPR_STMT);
    node->expr = expr;
    return (nc_ast_node*)node;
}

static nc_ast_node* statement(nc_parser* parser);

static nc_ast_node* block(nc_parser* parser) {
    ALLOC_NODE(nc_ast_block, AST_BLOCK);
    node->statements = NULL;
    node->n_stmts = 0;
    
    // Dynamic array for statements
    size_t capacity = 0;
    
    while (!match_token(parser, TOKEN_RBRACE) && !match_token(parser, TOKEN_EOF)) {
        nc_ast_node* stmt = declaration(parser);
        if (stmt) {
            if (capacity < node->n_stmts + 1) {
                capacity = (capacity < 8) ? 8 : capacity * 2;
                node->statements = (nc_ast_node**)realloc(node->statements, sizeof(nc_ast_node*) * capacity);
            }
            node->statements[node->n_stmts++] = stmt;
        }
    }
    
    return (nc_ast_node*)node;
}

static nc_ast_node* if_statement(nc_parser* parser) {
    consume(parser, TOKEN_LPAREN, "Expect '(' after 'if'.");
    nc_ast_node* condition = expression(parser);
    consume(parser, TOKEN_RPAREN, "Expect ')' after if condition.");
    
    nc_ast_node* then_branch = statement(parser);
    nc_ast_node* else_branch = NULL;
    
    if (match_token(parser, TOKEN_ELSE)) {
        else_branch = statement(parser);
    }
    
    ALLOC_NODE(nc_ast_if, AST_IF);
    node->condition = condition;
    node->then_branch = then_branch;
    node->else_branch = else_branch;
    return (nc_ast_node*)node;
}

static nc_ast_node* while_statement(nc_parser* parser) {
    consume(parser, TOKEN_LPAREN, "Expect '(' after 'while'.");
    nc_ast_node* condition = expression(parser);
    consume(parser, TOKEN_RPAREN, "Expect ')' after while condition.");
    
    nc_ast_node* body = statement(parser);
    
    ALLOC_NODE(nc_ast_while, AST_WHILE);
    node->condition = condition;
    node->body = body;
    return (nc_ast_node*)node;
}

static nc_ast_node* for_statement(nc_parser* parser) {
    consume(parser, TOKEN_LPAREN, "Expect '(' after 'for'.");
    consume(parser, TOKEN_VAR, "Expect 'var' in for-in loop."); // MVP restriction
    consume(parser, TOKEN_IDENTIFIER, "Expect variable name.");
    nc_token var_name = parser->previous;
    
    consume(parser, TOKEN_IN, "Expect 'in' after variable name.");
    nc_ast_node* iterable = expression(parser);
    consume(parser, TOKEN_RPAREN, "Expect ')' after for clauses.");
    
    nc_ast_node* body = statement(parser);
    
    ALLOC_NODE(nc_ast_for_in, AST_FOR_IN);
    node->var_name = var_name;
    node->iterable = iterable;
    node->body = body;
    return (nc_ast_node*)node;
}

static nc_ast_node* break_statement(nc_parser* parser) {
    consume(parser, TOKEN_SEMICOLON, "Expect ';' after 'break'.");
    ALLOC_NODE(nc_ast_break, AST_BREAK);
    return (nc_ast_node*)node;
}

static nc_ast_node* continue_statement(nc_parser* parser) {
    consume(parser, TOKEN_SEMICOLON, "Expect ';' after 'continue'.");
    ALLOC_NODE(nc_ast_continue, AST_CONTINUE);
    return (nc_ast_node*)node;
}

static nc_ast_node* return_statement(nc_parser* parser) {
    nc_ast_node* value = NULL;
    if (!match_token(parser, TOKEN_SEMICOLON)) {
        value = expression(parser);
        consume(parser, TOKEN_SEMICOLON, "Expect ';' after return value.");
    }
    
    ALLOC_NODE(nc_ast_return, AST_RETURN);
    node->value = value;
    return (nc_ast_node*)node;
}

static nc_ast_node* function_declaration(nc_parser* parser, nc_token type) {
    // We consumed Type. Current is Identifier.
    consume(parser, TOKEN_IDENTIFIER, "Expect function name.");
    nc_token name = parser->previous;
    
    consume(parser, TOKEN_LPAREN, "Expect '(' after function name.");
    
    // Parse parameters
    nc_token* params = NULL;
    size_t param_count = 0;
    size_t param_cap = 0;
    
    if (parser->current.type != TOKEN_RPAREN) {
        do {
            if (param_count >= 255) {
                error_at(parser, &parser->current, "Can't have more than 255 parameters.");
            }
            
            // Param Type (int, float, etc) - ignored for MVP runtime but parsed
            if (match_token(parser, TOKEN_INT) || match_token(parser, TOKEN_FLOAT) || 
                match_token(parser, TOKEN_STRING) || match_token(parser, TOKEN_BOOL) ||
                match_token(parser, TOKEN_VAR) || match_token(parser, TOKEN_TENSOR)) {
                // Consumed type
                // fprintf(stderr, "Consumed parameter type.\n");
            } else {
                // fprintf(stderr, "No parameter type found. Next: %d\n", parser->current.type);
            }
            
            consume(parser, TOKEN_IDENTIFIER, "Expect parameter name.");
            // fprintf(stderr, "Consumed parameter name: %.*s\n", parser->previous.length, parser->previous.start);
            
            if (param_cap < param_count + 1) {
                param_cap = (param_cap < 8) ? 8 : param_cap * 2;
                params = realloc(params, sizeof(nc_token) * param_cap);
            }
            params[param_count++] = parser->previous;
        } while (match_token(parser, TOKEN_COMMA));
    }
    
    consume(parser, TOKEN_RPAREN, "Expect ')' after parameters.");
    
    consume(parser, TOKEN_LBRACE, "Expect '{' before function body.");
    nc_ast_node* body = block(parser);
    
    ALLOC_NODE(nc_ast_function, AST_FUNCTION);
    node->name = name;
    node->params = params;
    node->param_count = param_count;
    node->body = body;
    return (nc_ast_node*)node;
}

static nc_ast_node* declaration(nc_parser* parser);

static nc_ast_node* class_declaration(nc_parser* parser) {
    consume(parser, TOKEN_IDENTIFIER, "Expect class name.");
    nc_token name = parser->previous;
    consume(parser, TOKEN_LBRACE, "Expect '{' before class body.");
    
    nc_ast_node** methods = NULL;
    size_t method_count = 0;
    size_t method_cap = 0;

    nc_ast_node** members = NULL;
    size_t member_count = 0;
    size_t member_cap = 0;
    
    while (!match_token(parser, TOKEN_RBRACE) && !match_token(parser, TOKEN_EOF)) {
        // Reuse declaration parser.
        // It returns AST_VAR_DECL for fields, AST_FUNCTION for methods.
        // Note: declaration() handles TYPE detection.
        nc_ast_node* member = declaration(parser);
        
        if (member) {
            if (member->type == AST_FUNCTION) {
                if (method_count + 1 > method_cap) {
                    method_cap = (method_cap < 8) ? 8 : method_cap * 2;
                    methods = realloc(methods, sizeof(nc_ast_node*) * method_cap);
                }
                methods[method_count++] = member;
            } else if (member->type == AST_VAR_DECL) {
                if (member_count + 1 > member_cap) {
                    member_cap = (member_cap < 8) ? 8 : member_cap * 2;
                    members = realloc(members, sizeof(nc_ast_node*) * member_cap);
                }
                members[member_count++] = member;
            } else {
                 // Ignore or error?
                 // declaration might return other things if we expand syntax.
                 nc_ast_free(member);
            }
        }
    }
    
    ALLOC_NODE(nc_ast_class_decl, AST_CLASS_DECL);
    node->name = name;
    node->members = members;
    node->member_count = member_count;
    node->methods = methods;
    node->method_count = method_count;
    
    return (nc_ast_node*)node;
}

static nc_ast_node* declaration(nc_parser* parser) {
    if (match_token(parser, TOKEN_CLASS)) {
        return class_declaration(parser);
    }
    
    // Check for types
    if (match_token(parser, TOKEN_VAR) || 
        match_token(parser, TOKEN_INT) || 
        match_token(parser, TOKEN_FLOAT) || 
        match_token(parser, TOKEN_STRING) ||
        match_token(parser, TOKEN_BOOL) ||
        match_token(parser, TOKEN_TENSOR) ||
        match_token(parser, TOKEN_VOID)) {
        
        nc_token type = parser->previous;
        
        // Lookahead to distinguish var decl from func decl
        // Func: Type Identifier LPAREN
        // Var:  Type Identifier (EQUAL or SEMICOLON or LBRACKET/LESS for tensor generics)
        
        if (parser->current.type == TOKEN_IDENTIFIER) {
            // Peek at next token using lexer? 
            // Our lexer doesn't support peeking easily without consuming state or using `parser->current` + next.
            // But we only validly have `parser->previous` and `parser->current`.
            // We need to peek one more.
            // HACK: We can scan one token, check it, and if not LPAREN, we are in trouble if we can't unget.
            // Alternative: Parser Lookahead.
            
            // Currently `nc_parser` struct has `current` and `previous`.
            // Let's implement specific check.
            
            // If we assume valid syntax:
            // int x = ...
            // int x;
            // int foo(...)
            
            // If we are at Identifier (x or foo).
            // We can consume identifier.
            // Then check if next is LPAREN.
            
            consume(parser, TOKEN_IDENTIFIER, "Expect name.");
            nc_token name = parser->previous;
            
            if (match_token(parser, TOKEN_LPAREN)) {
                // It is a function!
                // But wait, `function_declaration` helper expects to handle parsing parameters etc.
                // and I just consumed the Name and LPAREN.
                // I need to adapt `function_declaration` or inline logic here.
                
                // Let's Inline or refactor helper.
                
                // Parse params (Common logic)
                nc_token* params = NULL;
                size_t param_count = 0;
                size_t param_cap = 0;
                
                if (parser->current.type != TOKEN_RPAREN) {
                    do {
                         // Param parsing... (Same as above)
                        if (match_token(parser, TOKEN_INT) || match_token(parser, TOKEN_FLOAT) || 
                            match_token(parser, TOKEN_STRING) || match_token(parser, TOKEN_BOOL) ||
                            match_token(parser, TOKEN_VAR) || match_token(parser, TOKEN_TENSOR)) {
                        }
                        
                        consume(parser, TOKEN_IDENTIFIER, "Expect parameter name.");
                        if (param_cap < param_count + 1) {
                            param_cap = (param_cap < 8) ? 8 : param_cap * 2;
                            params = realloc(params, sizeof(nc_token) * param_cap);
                        }
                        params[param_count++] = parser->previous;
                    } while (match_token(parser, TOKEN_COMMA));
                }
                consume(parser, TOKEN_RPAREN, "Expect ')' after parameters.");
                
                consume(parser, TOKEN_LBRACE, "Expect '{' before function body.");
                nc_ast_node* body = block(parser);
                
                ALLOC_NODE(nc_ast_function, AST_FUNCTION);
                node->name = name;
                node->params = params;
                node->param_count = param_count;
                node->body = body;
                return (nc_ast_node*)node;

            } else {
                // It MUST be a variable declaration
                // We typically handle TENSOR<...> parsing in var_declaration.
                // But we already consumed type.
                
                // Reconstruct var_declaration logic starting after type/name?
                // `var_declaration` expects to consume Identifier.
                // We consumed Identifier `name`.
                
                nc_ast_node* initializer = NULL;
                if (match_token(parser, TOKEN_EQUAL)) {
                    initializer = expression(parser);
                }
                
                consume(parser, TOKEN_SEMICOLON, "Expect ';' after variable declaration.");
                
                ALLOC_NODE(nc_ast_var_decl, AST_VAR_DECL);
                node->type_tok = type;
                node->name = name;
                node->initializer = initializer;
                return (nc_ast_node*)node;
            }
        }
    }
    
    return statement(parser);
}

static nc_ast_node* statement(nc_parser* parser) {
    if (match_token(parser, TOKEN_FOR)) {
        return for_statement(parser);
    }
    if (match_token(parser, TOKEN_IF)) {
        return if_statement(parser);
    }
    if (match_token(parser, TOKEN_WHILE)) {
        return while_statement(parser);
    }
    if (match_token(parser, TOKEN_BREAK)) {
        return break_statement(parser);
    }
    if (match_token(parser, TOKEN_CONTINUE)) {
        return continue_statement(parser);
    }
    if (match_token(parser, TOKEN_RETURN)) {
        return return_statement(parser);
    }
    if (match_token(parser, TOKEN_LBRACE)) {
        return block(parser);
    }
    
    // We shouldn't handle VAR/INT here anymore if we want top-level to handle them via declaration()
    // But block() calls statement(). Blocks can have decls.
    // So statement() should probably delegate to declaration() if types match?
    // Or declaration() calls statement() if no type matched?
    // "declaration" encompasses both vars/funcs AND statements.
    
    return expression_statement(parser);
}

// ============================================================
// Top Level
// ============================================================

nc_ast_program* nc_parser_parse(nc_parser* parser) {
    ALLOC_NODE(nc_ast_program, AST_PROGRAM);
    node->statements = malloc(sizeof(nc_ast_node*) * 64); // Arbitrary limit for draft
    node->n_stmts = 0;
    
    while (parser->current.type != TOKEN_EOF) {
        nc_ast_node* stmt = declaration(parser);
        if (stmt) {
            node->statements[node->n_stmts++] = stmt;
        }
    }
    
    return node;
}
