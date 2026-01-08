#ifndef NOCTA_LANG_AST_H
#define NOCTA_LANG_AST_H

#include "token.h"

typedef enum {
    // Expressions
    AST_LITERAL,
    AST_BINARY,
    AST_LOGICAL,
    AST_UNARY,
    AST_VAR_ACCESS,
    AST_ASSIGNMENT,
    AST_CALL,
    AST_MEMBER_ACCESS, // for obj.method()

    // Statements
    AST_EXPR_STMT,
    AST_VAR_DECL,
    AST_BLOCK,
    AST_IF,
    AST_WHILE,
    AST_FUNCTION,
    AST_RETURN,
    AST_FOR_IN,
    AST_CLASS_DECL,
    AST_SET_MEMBER,
    AST_LIST_LITERAL,
    AST_INDEX_ACCESS,
    AST_INDEX_SET,
    AST_BREAK,
    AST_CONTINUE,
    
    AST_PROGRAM // Root node
} nc_ast_node_type;

typedef struct nc_ast_node nc_ast_node;

// Base structure for checking type
struct nc_ast_node {
    nc_ast_node_type type;
};

// ============================================
// Expressions
// ============================================

typedef struct {
    nc_ast_node base;
    nc_token token; // The number or string literal token
} nc_ast_literal;

typedef struct {
    nc_ast_node base;
    nc_token operator;
    nc_ast_node* left;
    nc_ast_node* right;
} nc_ast_binary;

typedef struct {
    nc_ast_node base;
    nc_token operator;
    nc_ast_node* left;
    nc_ast_node* right;
} nc_ast_logical;

typedef struct {
    nc_ast_node base;
    nc_token operator;
    nc_ast_node* right;
} nc_ast_unary;

typedef struct {
    nc_ast_node base;
    nc_token name;
} nc_ast_var_access;

typedef struct {
    nc_ast_node base;
    nc_token name;      // Variable name being assigned to
    nc_ast_node* value; // Expression value
} nc_ast_assignment;

typedef struct {
    nc_ast_node base;
    nc_ast_node* callee; // Function being called (usually VAR_ACCESS or MEMBER_ACCESS)
    nc_ast_node** args;  // Array of arguments
    size_t n_args;
} nc_ast_call;

typedef struct {
    nc_ast_node base;
    nc_ast_node* object;
    nc_token member;
} nc_ast_member_access;

typedef struct {
    nc_ast_node base;
    nc_ast_node* object;
    nc_token member;
    nc_token operator; // For compound assignment (TOKEN_PLUS_EQUAL etc, or TOKEN_EQUAL/TOKEN_ERROR if simple)
    nc_ast_node* value;
} nc_ast_set_member;

// ============================================
// Statements
// ============================================

typedef struct {
    nc_ast_node base;
    nc_ast_node* expr;
} nc_ast_expr_stmt;

typedef struct {
    nc_ast_node base;
    nc_ast_node** elements;
    size_t element_count;
} nc_ast_list_literal;

typedef struct {
    nc_ast_node base;
    nc_ast_node* object;
    nc_ast_node* index;
} nc_ast_index_access;

typedef struct {
    nc_ast_node base;
    nc_ast_node* object;
    nc_ast_node* index;
    nc_ast_node* value;
} nc_ast_index_set;

typedef struct {
    nc_ast_node base;
    nc_token type_tok; // e.g. "int", "var", "tensor"
    nc_token name;
    nc_ast_node* initializer; // Optional, can be NULL
} nc_ast_var_decl;

typedef struct {
    nc_ast_node base;
    nc_token name;
    nc_token* params;     // Dynamic array of param names
    size_t param_count;
    nc_ast_node* body;    // AST_BLOCK
} nc_ast_function;

typedef struct {
    nc_ast_node base;
    nc_ast_node* value;   // Optional return value (NULL if void)
} nc_ast_return;

typedef struct {
    nc_ast_node base;
} nc_ast_break;

typedef struct {
    nc_ast_node base;
} nc_ast_continue;

typedef struct {
    nc_ast_node base;
    nc_ast_node** statements; // Array of top-level statements
    size_t n_stmts;
} nc_ast_program;

typedef struct {
    nc_ast_node base;
    nc_ast_node** statements; // Array of statements
    size_t n_stmts;
} nc_ast_block;

typedef struct {
    nc_ast_node base;
    nc_ast_node* condition;
    nc_ast_node* then_branch;
    nc_ast_node* else_branch; // Optional, can be NULL
} nc_ast_if;

typedef struct {
    nc_ast_node base;
    nc_ast_node* condition;
    nc_ast_node* body;
} nc_ast_while;

typedef struct {
    nc_ast_node base;
    nc_token var_name;
    nc_ast_node* iterable;
    nc_ast_node* body;
} nc_ast_for_in;

typedef struct {
    nc_ast_node base;
    nc_token name;
    // For MVP, class body contains variable declarations? 
    // Or method definitions?
    // Let's assume statements in body, interpreted as members?
    // Let's treat body as block for now.
    nc_ast_node** members; // Var declarations
    size_t member_count;
    
    nc_ast_node** methods; // Function declarations
    size_t method_count;
} nc_ast_class_decl;

typedef struct {
    nc_ast_node base;
    nc_token return_type;
    nc_token name;
    // Parameters (simplified for now, usually needs a struct)
    // For now assuming list of VarDecls or similar
    nc_token* param_types; 
    nc_token* param_names;
    size_t n_params;
    
    nc_ast_node* body; // Block
} nc_ast_func_decl;

// Function prototypes
void nc_ast_free(nc_ast_node* node);

#endif // NOCTA_LANG_AST_H
