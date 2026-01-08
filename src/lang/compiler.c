#include "nocta/lang/compiler.h"
#include "nocta/lang/object.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Compiler Context & State
typedef struct {
    nc_token name;
    int depth;
} nc_local;

typedef enum {
    TYPE_FUNCTION,
    TYPE_SCRIPT,
    TYPE_METHOD
} FunctionType;

typedef struct Loop {
    struct Loop* enclosing;
    int start;
    int scope_depth;
    int local_count;
    
    // Breaks to patch
    // Simple fixed buffer for MVP
    int break_jumps[256];
    int break_count;
} Loop;

typedef struct nc_compiler {
    struct nc_compiler* enclosing;
    nc_function* function;
    FunctionType type;
    
    nc_local locals[256];
    int local_count;
    int scope_depth;
    
    Loop* loop;
} nc_compiler;

static nc_compiler* current = NULL;
static bool had_error = false;

static nc_chunk* current_chunk() {
    return &current->function->chunk;
}

static void emit_byte(uint8_t byte) {
    nc_chunk_write(current_chunk(), byte, 0); 
}

static void emit_bytes(uint8_t byte1, uint8_t byte2) {
    emit_byte(byte1);
    emit_byte(byte2);
}

static void emit_return() {
    emit_byte(OP_NIL); // Default return nil
    emit_byte(OP_RETURN);
}

static void emit_constant(nc_value value) {
    int const_idx = nc_chunk_add_constant(current_chunk(), value);
    if (const_idx > 255) {
        fprintf(stderr, "Too many constants in chunk\n");
        had_error = true;
        return;
    }
    emit_bytes(OP_CONSTANT, (uint8_t)const_idx);
}

static void init_compiler(nc_compiler* compiler, FunctionType type) {
    compiler->enclosing = current;
    compiler->function = NULL;
    compiler->type = type;
    compiler->local_count = 0;
    compiler->scope_depth = 0;
    compiler->loop = NULL;
    
    compiler->function = nc_new_function();
    current = compiler;
    
    // Reserve slot 0 for function name/this
    if (type != TYPE_SCRIPT) {
        compiler->function->name = nc_allocate_string("", 0); // Placeholder
    }
    
    nc_local* local = &current->locals[current->local_count++];
    local->depth = 0;
    if (type == TYPE_METHOD) {
        local->name.start = "this";
        local->name.length = 4;
    } else {
        local->name.start = "";
        local->name.length = 0;
    }
}

static nc_function* end_compiler() {
    emit_return();
    nc_function* function = current->function;
    current = current->enclosing;
    return function;
}

static void begin_scope() {
    current->scope_depth++;
}

static void end_scope() {
    current->scope_depth--;
    
    // Pop locals
    while (current->local_count > 0 && 
           current->locals[current->local_count - 1].depth > current->scope_depth) {
        emit_byte(OP_POP);
        current->local_count--;
    }
}

// Forward decl
static void compile_node(nc_ast_node* node);
static void compile_var_access(nc_ast_var_access* node);
static void compile_assignment(nc_ast_assignment* node);
static void compile_function(nc_ast_function* node, FunctionType type);
static void compile_return(nc_ast_return* node);


// Variable resolution
static int resolve_local(nc_compiler* compiler, nc_token* name) {
    for (int i = compiler->local_count - 1; i >= 0; i--) {
        nc_local* local = &compiler->locals[i];
        if (local->name.length == name->length &&
            memcmp(local->name.start, name->start, name->length) == 0) {
            return i;
        }
    }
    return -1;
}

static void add_local(nc_token name) {
    if (current->local_count == 256) {
        fprintf(stderr, "Too many local variables in function.");
        return;
    }
    
    nc_local* local = &current->locals[current->local_count++];
    local->name = name;
    local->depth = current->scope_depth;
}

// String copying for constant pool (MVP: leaking memory or simple duplication)
static nc_value make_string_val(const char* chars, size_t length) {
    return OBJ_VAL(nc_allocate_string(chars, (int)length));
}

static void compile_binary(nc_ast_binary* node) {
    compile_node(node->left);
    compile_node(node->right);
    
    switch (node->operator.type) {
        case TOKEN_PLUS:  emit_byte(OP_ADD); break;
        case TOKEN_MINUS: emit_byte(OP_SUB); break;
        case TOKEN_STAR:  emit_byte(OP_MUL); break;
        case TOKEN_SLASH: emit_byte(OP_DIV); break;
        case TOKEN_PERCENT: emit_byte(OP_MOD); break;
        
        case TOKEN_EQ: emit_byte(OP_EQUAL); break;
        case TOKEN_NE:    emit_bytes(OP_EQUAL, OP_NOT); break;
        case TOKEN_GT:    emit_byte(OP_GREATER); break;
        case TOKEN_GE:    emit_bytes(OP_LESS, OP_NOT); break;
        case TOKEN_LT:    emit_byte(OP_LESS); break;
        case TOKEN_LE:    emit_bytes(OP_GREATER, OP_NOT); break;
        // ... other ops
        default: break;
    }
}

static void compile_unary(nc_ast_unary* node) {
    compile_node(node->right);
    switch (node->operator.type) {
        case TOKEN_MINUS: emit_byte(OP_NEGATE); break;
        case TOKEN_NOT: emit_byte(OP_NOT); break;
        default: break;
    }
}

static void compile_literal(nc_ast_literal* node) {
    switch (node->token.type) {
        case TOKEN_NUMBER: {
            double value = strtod(node->token.start, NULL);
            emit_constant(NUMBER_VAL(value));
            break;
        }
        case TOKEN_TRUE: emit_byte(OP_TRUE); break;
        case TOKEN_FALSE: emit_byte(OP_FALSE); break;
        case TOKEN_NIL: emit_byte(OP_NIL); break;
        case TOKEN_STRING_LIT: {
            // Remove quotes
            // Note: node->token.start points to opening quote "
            // len includes quotes.
            // We want text inside.
            nc_value value = make_string_val(node->token.start + 1, node->token.length - 2);
            emit_constant(value);
            break;
        }
        default: break;
    }
}

static void compile_var_decl(nc_ast_var_decl* node) {
    // 1. Compile initializer
    if (node->initializer) {
        compile_node(node->initializer);
    } else {
        emit_byte(OP_NIL);
    }
    
    // 2. Define Variable
    if (current->scope_depth > 0) {
        add_local(node->name);
    } else {
        // Global
        nc_value nameVal = make_string_val(node->name.start, node->name.length);
        int arg = nc_chunk_add_constant(current_chunk(), nameVal);
        emit_bytes(OP_DEFINE_GLOBAL, (uint8_t)arg);
    }
}

static void compile_expr_stmt(nc_ast_expr_stmt* node) {
    compile_node(node->expr);
    emit_byte(OP_POP); // Expression statement result is discarded
}

// --- Control Flow Helpers ---

static int emit_jump(uint8_t instruction) {
    emit_byte(instruction);
    emit_byte(0xff); // Placeholder
    emit_byte(0xff); // Placeholder
    return current_chunk()->count - 2;
}

static void patch_jump(int offset) {
    // -2 to adjust for the jump offset itself
    int jump = current_chunk()->count - offset - 2;
    
    if (jump > UINT16_MAX) {
        fprintf(stderr, "Too much code to jump over.\n");
        had_error = true;
    }
    
    current_chunk()->code[offset] = (jump >> 8) & 0xff;
    current_chunk()->code[offset + 1] = jump & 0xff;
}

static void emit_loop(int loop_start) {
    emit_byte(OP_LOOP);
    
    int offset = current_chunk()->count - loop_start + 2;
    if (offset > UINT16_MAX) had_error = true;
    
    emit_byte((offset >> 8) & 0xff);
    emit_byte(offset & 0xff);
}

// compile_block logic was already updated in previous step to call begin/end scope
// BUT previous step might have missed the definition if it wasn't in the chunk range.
// Let's redefine it to be safe.

static void compile_block(nc_ast_block* node) {
    begin_scope();
    for (size_t i = 0; i < node->n_stmts; i++) {
        compile_node(node->statements[i]);
    }
    end_scope();
}

static void compile_break(nc_ast_break* node) {
    if (current->loop == NULL) {
        fprintf(stderr, "Cannot break outside of loop.\n");
        had_error = true;
        return;
    }
    
    int discard = current->local_count - current->loop->local_count;
    for (int i = 0; i < discard; i++) {
        emit_byte(OP_POP);
    }
    
    if (current->loop->break_count < 256) {
        current->loop->break_jumps[current->loop->break_count++] = emit_jump(OP_JUMP);
    } else {
        fprintf(stderr, "Too many breaks in loop.\n");
        had_error = true;
    }
}

static void compile_continue(nc_ast_continue* node) {
    if (current->loop == NULL) {
        fprintf(stderr, "Cannot continue outside of loop.\n");
        had_error = true;
        return;
    }
    
    int discard = current->local_count - current->loop->local_count;
    for (int i = 0; i < discard; i++) {
        emit_byte(OP_POP);
    }
    
    emit_loop(current->loop->start);
}

static void compile_if(nc_ast_if* node) {
    compile_node(node->condition);
    
    // Jump to else if false
    int then_jump = emit_jump(OP_JUMP_IF_FALSE);
    emit_byte(OP_POP); // Pop condition
    
    compile_node(node->then_branch);
    
    int else_jump = emit_jump(OP_JUMP);
    
    patch_jump(then_jump);
    emit_byte(OP_POP); 
    
    if (node->else_branch) {
        compile_node(node->else_branch);
    }
    patch_jump(else_jump);
}

static void compile_while(nc_ast_while* node) {
    int loop_start = current_chunk()->count;
    
    // Loop Context
    Loop loop;
    loop.start = loop_start;
    loop.scope_depth = current->scope_depth;
    loop.local_count = current->local_count;
    loop.enclosing = current->loop;
    loop.break_count = 0;
    current->loop = &loop;
    
    compile_node(node->condition);
    
    int exit_jump = emit_jump(OP_JUMP_IF_FALSE);
    emit_byte(OP_POP);
    
    compile_node(node->body);
    
    emit_loop(loop_start);
    
    patch_jump(exit_jump);
    emit_byte(OP_POP);
    
    // Restore loop context
    current->loop = loop.enclosing;
    
    // Patch breaks
    for (int i = 0; i < loop.break_count; i++) {
        patch_jump(loop.break_jumps[i]);
    }
}

static void compile_call(nc_ast_call* node) {
    // 1. Compile Callee
    compile_node(node->callee);
    
    // 2. Compile Arguments
    for (size_t i = 0; i < node->n_args; i++) {
        compile_node(node->args[i]);
    }
    
    // 3. Emit OP_CALL
    emit_bytes(OP_CALL, (uint8_t)node->n_args);
}

static void compile_var_access(nc_ast_var_access* node) {
    int arg = resolve_local(current, &node->name);
    if (arg != -1) {
        emit_bytes(OP_GET_LOCAL, (uint8_t)arg);
    } else {
        nc_value nameVal = make_string_val(node->name.start, node->name.length);
        arg = nc_chunk_add_constant(current_chunk(), nameVal);
        emit_bytes(OP_GET_GLOBAL, (uint8_t)arg);
    }
}

static void compile_assignment(nc_ast_assignment* node) {
    compile_node(node->value);
    
    int arg = resolve_local(current, &node->name);
    if (arg != -1) {
        emit_bytes(OP_SET_LOCAL, (uint8_t)arg);
    } else {
        nc_value nameVal = make_string_val(node->name.start, node->name.length);
        arg = nc_chunk_add_constant(current_chunk(), nameVal);
        emit_bytes(OP_SET_GLOBAL, (uint8_t)arg);
    }
}

static void compile_function(nc_ast_function* node, FunctionType type) {
    nc_compiler compiler;
    init_compiler(&compiler, type);
    
    // Function Name
    compiler.function->name = nc_allocate_string(node->name.start, (int)node->name.length);
    compiler.function->arity = (int)node->param_count; 
    
    begin_scope();
    // Define parameters as locals
    for (size_t i = 0; i < node->param_count; i++) {
        add_local(node->params[i]);
    }
    
    compile_node(node->body);
    
    nc_function* function = end_compiler();
    
    // Emit closure in enclosing compiler
    emit_constant(OBJ_VAL(function));

    if (type == TYPE_METHOD) return;

    if (current->scope_depth > 0) {
        add_local(node->name);
    } else {
        nc_value nameVal = make_string_val(node->name.start, node->name.length);
        int arg = nc_chunk_add_constant(current_chunk(), nameVal);
        emit_bytes(OP_DEFINE_GLOBAL, (uint8_t)arg);
    }
}

static void compile_return(nc_ast_return* node) {
    if (current->type == TYPE_SCRIPT) {
        fprintf(stderr, "Can't return from top-level code.");
        // had_error = true; 
    }
    
    if (node->value) {
        compile_node(node->value);
        emit_byte(OP_RETURN);
    } else {
        emit_return(); // Returns nil
    }
}

static void compile_for_in(nc_ast_for_in* node) {
    // 1. Compile iterable (pushes range/iterator object)
    compile_node(node->iterable);
    
    // 1b. Reserve stack slot for iterator (hidden local)
    // allowing subsequent locals to be mapped correctly
    nc_token iter_token;
    iter_token.start = "";
    iter_token.length = 0;
    add_local(iter_token);
    
    // 2. Loop Start Label
    int loop_start = current_chunk()->count;
    
    // Loop Context
    Loop loop;
    loop.start = loop_start;
    loop.scope_depth = current->scope_depth;
    loop.local_count = current->local_count;
    loop.enclosing = current->loop;
    loop.break_count = 0;
    current->loop = &loop;
    
    // 3. Emit OP_FOR_ITER [exit_jump]
    // Consumes nothing (peeks iterator), Pushes Value if valid.
    int exit_jump = emit_jump(OP_FOR_ITER);
    
    // 4. Scope for Loop Variable
    begin_scope();
    add_local(node->var_name); // The value pushed by FOR_ITER is now bound to this local
    
    // 5. Compile Body
    compile_node(node->body);
    
    // 6. End Scope (Pop loop variable)
    end_scope();
    
    // 7. Loop back
    emit_loop(loop_start);
    
    // Restore loop context
    current->loop = loop.enclosing;
    
    // 8. Patch Exit
    patch_jump(exit_jump);
    
    // Patch breaks (they jump to here, to pop iterator)
    for (int i = 0; i < loop.break_count; i++) {
        patch_jump(loop.break_jumps[i]);
    }
    
    // 9. Pop Iterator (Stack cleanup)
    emit_byte(OP_POP);
    
    // 10. Sync compiler state: Remove hidden iterator local
    current->local_count--;
}

static void compile_class_decl(nc_ast_class_decl* node) {
    // 1. Define class name constant
    nc_value nameVal = make_string_val(node->name.start, node->name.length);
    int const_idx = nc_chunk_add_constant(current_chunk(), nameVal);
    
    // 2. Emit OP_CLASS
    emit_bytes(OP_CLASS, (uint8_t)const_idx);
    
    // 3. Define Global Variable for the class
    int name_idx = nc_chunk_add_constant(current_chunk(), nameVal);
    emit_bytes(OP_DEFINE_GLOBAL, (uint8_t)name_idx);
    
    // 4. Push Class back onto stack for method binding
    emit_bytes(OP_GET_GLOBAL, (uint8_t)name_idx);
    
    // 5. Compile Methods
    for (size_t i = 0; i < node->method_count; i++) {
        nc_ast_node* methodNode = node->methods[i];
        if (methodNode->type == AST_FUNCTION) {
            nc_ast_function* func = (nc_ast_function*)methodNode;
            compile_function(func, TYPE_METHOD);
            
            nc_value methodName = make_string_val(func->name.start, func->name.length);
            int m_idx = nc_chunk_add_constant(current_chunk(), methodName);
            emit_bytes(OP_METHOD, (uint8_t)m_idx);
        }
    }
    
    // 6. Pop Class
    emit_byte(OP_POP);
}

static void compile_member_access(nc_ast_member_access* node) {
    compile_node(node->object);
    
    nc_value nameVal = make_string_val(node->member.start, node->member.length);
    int arg = nc_chunk_add_constant(current_chunk(), nameVal);
    emit_bytes(OP_GET_PROPERTY, (uint8_t)arg);
}

static void compile_set_member(nc_ast_set_member* node) {
    compile_node(node->object);
    
    // Handle Compound Assignment
    if (node->operator.type != TOKEN_EQUAL && node->operator.type != TOKEN_ERROR) {
        // Compound: obj.x += val
        // Stack: [obj]
        emit_byte(OP_DUP); // [obj, obj]
        
        nc_value nameVal = make_string_val(node->member.start, node->member.length);
        int arg = nc_chunk_add_constant(current_chunk(), nameVal);
        emit_bytes(OP_GET_PROPERTY, (uint8_t)arg); // [obj, old_val]
        
        compile_node(node->value); // [obj, old_val, rhs]
        
        switch (node->operator.type) {
            case TOKEN_PLUS_EQUAL:  emit_byte(OP_ADD); break;
            case TOKEN_MINUS_EQUAL: emit_byte(OP_SUB); break;
            case TOKEN_STAR_EQUAL:  emit_byte(OP_MUL); break;
            case TOKEN_SLASH_EQUAL: emit_byte(OP_DIV); break;
            default: break; // Should not happen
        }
        // Stack: [obj, result]
        
        // Use same constant index for Setter
        emit_bytes(OP_SET_PROPERTY, (uint8_t)arg);
    } else {
        // Simple Assignment: obj.x = val
        compile_node(node->value);
        // Stack: [obj, val]
        
        nc_value nameVal = make_string_val(node->member.start, node->member.length);
        int arg = nc_chunk_add_constant(current_chunk(), nameVal);
        emit_bytes(OP_SET_PROPERTY, (uint8_t)arg);
    }
}

static void compile_list_literal(nc_ast_list_literal* node) {
    for (size_t i = 0; i < node->element_count; i++) {
        compile_node(node->elements[i]);
    }
    emit_bytes(OP_BUILD_LIST, (uint8_t)node->element_count);
}

static void compile_index_access(nc_ast_index_access* node) {
    compile_node(node->object);
    compile_node(node->index);
    emit_byte(OP_GET_INDEX);
}

static void compile_logical(nc_ast_logical* node) {
    compile_node(node->left);
    
    if (node->operator.type == TOKEN_AND) {
        // AND: If false, short circuit (jump to end, result is false)
        int end_jump = emit_jump(OP_JUMP_IF_FALSE);
        emit_byte(OP_POP); // Left was true, discard
        compile_node(node->right);
        patch_jump(end_jump);
        // Stack: [result_of_right] (if eval'd) OR [false] (if jumped)
    } else {
        // OR: If false, jump to right eval. If true, jump to end (result is true)
        int else_jump = emit_jump(OP_JUMP_IF_FALSE);
        int end_jump = emit_jump(OP_JUMP);
        
        patch_jump(else_jump);
        emit_byte(OP_POP); // Left was false, discard
        compile_node(node->right);
        
        patch_jump(end_jump);
    }
}

static void compile_index_set(nc_ast_index_set* node) {
    compile_node(node->object);
    compile_node(node->index);
    compile_node(node->value);
    emit_byte(OP_SET_INDEX);
}

static void compile_node(nc_ast_node* node) {
    if (!node) return;
    switch (node->type) {
        case AST_BINARY: compile_binary((nc_ast_binary*)node); break;
        case AST_LOGICAL: compile_logical((nc_ast_logical*)node); break;
        case AST_UNARY: compile_unary((nc_ast_unary*)node); break;
        case AST_LITERAL: compile_literal((nc_ast_literal*)node); break;
        case AST_VAR_DECL: compile_var_decl((nc_ast_var_decl*)node); break;
        case AST_EXPR_STMT: compile_expr_stmt((nc_ast_expr_stmt*)node); break;
        case AST_VAR_ACCESS: compile_var_access((nc_ast_var_access*)node); break;
        case AST_ASSIGNMENT: compile_assignment((nc_ast_assignment*)node); break;
        case AST_CALL: compile_call((nc_ast_call*)node); break;
        
        case AST_BLOCK: compile_block((nc_ast_block*)node); break;
        case AST_FUNCTION: compile_function((nc_ast_function*)node, TYPE_FUNCTION); break;
        case AST_RETURN: compile_return((nc_ast_return*)node); break;
        case AST_BREAK: compile_break((nc_ast_break*)node); break;
        case AST_CONTINUE: compile_continue((nc_ast_continue*)node); break;
        
        case AST_PROGRAM: {
            nc_ast_program* prog = (nc_ast_program*)node;
            for (size_t i = 0; i < prog->n_stmts; i++) {
                compile_node(prog->statements[i]);
            }
            break;
        }
        case AST_IF: compile_if((nc_ast_if*)node); break;
        case AST_WHILE: compile_while((nc_ast_while*)node); break;
        case AST_FOR_IN: compile_for_in((nc_ast_for_in*)node); break;
        
        case AST_CLASS_DECL: compile_class_decl((nc_ast_class_decl*)node); break;
        case AST_MEMBER_ACCESS: compile_member_access((nc_ast_member_access*)node); break;
        case AST_SET_MEMBER: compile_set_member((nc_ast_set_member*)node); break;
        
        case AST_LIST_LITERAL: compile_list_literal((nc_ast_list_literal*)node); break;
        case AST_INDEX_ACCESS: compile_index_access((nc_ast_index_access*)node); break;
        case AST_INDEX_SET: compile_index_set((nc_ast_index_set*)node); break;
        
        default: 
            fprintf(stderr, "Unknown or unsupported AST node type %d\n", node->type);
            break;
    }
}

// Global entry
bool nc_compile(nc_ast_program* program, nc_chunk* chunk) {
    nc_compiler compiler;
    init_compiler(&compiler, TYPE_SCRIPT);
    
    had_error = false;
    
    compile_node((nc_ast_node*)program);
    
    nc_function* script = end_compiler();
    
    if (had_error) return false;
    
    // Copy script chunk to output chunk
    // This is the bridge to existing API
    *chunk = script->chunk;
    
    return true;
}
