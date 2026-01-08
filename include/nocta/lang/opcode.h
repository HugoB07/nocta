#ifndef NOCTA_LANG_OPCODE_H
#define NOCTA_LANG_OPCODE_H

#include <stdint.h>

typedef enum {
    OP_CONSTANT,    // [const_index] Push constant to stack
    OP_NIL,         // Push nil
    OP_TRUE,        // Push true
    OP_FALSE,       // Push false
    OP_POP,         // Pop top of stack
    OP_DUP,         // Duplicate top of stack
    
    // Lists
    OP_BUILD_LIST,  // [item_count] Pop count items, push list
    OP_GET_INDEX,   // Pop index, list. Push item.
    OP_SET_INDEX,   // Pop value, index, list. Set item. Push value.
    
    // Variables
    OP_GET_LOCAL,   // [slot_index]
    OP_SET_LOCAL,   // [slot_index]
    OP_GET_GLOBAL,  // [name_const_index]
    OP_SET_GLOBAL,  // [name_const_index] // unused if we only support top-level scripts as main function
    OP_DEFINE_GLOBAL, // [name_const_index]
    
    // Functions
    OP_CALL,        // [arg_count]

    // Comparators
    OP_EQUAL,
    OP_GREATER,
    OP_LESS,

    // Binary Ops
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MOD,
    
    // Unary Ops
    OP_NOT,
    OP_NEGATE,

    // Control Flow
    OP_PRINT,       // Print top of stack (temporary for debug)
    OP_JUMP,        // [offset_short]
    OP_JUMP_IF_FALSE, // [offset_short]
    OP_LOOP,        // [offset_short] (jump back)
    OP_FOR_ITER,    // [offset_short] (check iter, jump exit if done, else push val)
    
    // Classes
    OP_CLASS,       // [name_const_index]
    OP_GET_PROPERTY,// [name_const_index]
    OP_SET_PROPERTY,// [name_const_index]
    OP_METHOD,      // [name_const_index]
    
    OP_RETURN,      // Return from function
    
} nc_opcode;

#endif // NOCTA_LANG_OPCODE_H
