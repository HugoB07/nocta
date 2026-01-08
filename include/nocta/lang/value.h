#ifndef NOCTA_LANG_VALUE_H
#define NOCTA_LANG_VALUE_H

#include <stdio.h>
#include <stdbool.h>
#include "nocta/core/tensor.h"

typedef enum {
    VAL_BOOL,
    VAL_NIL,
    VAL_NUMBER, // float/double
    VAL_TENSOR, // nc_tensor*
    VAL_NATIVE, // Native C function
    VAL_OBJ     // strings, functions, structs... (future)
} nc_value_type;

// typedef struct nc_obj nc_obj; // Forward decl or include object.h?
// It's cleaner to forward decl here or make generic void* casting inside macros.
// But we want type safety. object.h includes value.h so circular dependency.
// Solution: Forward declare nc_obj in value.h.

typedef struct nc_obj nc_obj;
typedef struct nc_value nc_value; // Forward decl
typedef struct nc_vm nc_vm; // Forward decl

typedef nc_value (*nc_native_fn)(nc_vm* vm, int arg_count, nc_value* args);

typedef struct nc_tensor_node {
    nc_tensor* tensor;
    bool is_marked;
    struct nc_tensor_node* next;
} nc_tensor_node;

struct nc_value {
    nc_value_type type;
    union {
        bool boolean;
        double number;
        nc_tensor_node* tensor_node; // Changed from nc_tensor*
        nc_native_fn native;
        nc_obj* obj;
    } as;
};

// Helper macros check type
#define IS_BOOL(v)    ((v).type == VAL_BOOL)
#define IS_NIL(v)     ((v).type == VAL_NIL)
#define IS_NUMBER(v)  ((v).type == VAL_NUMBER)
#define IS_TENSOR(v)  ((v).type == VAL_TENSOR)
#define IS_NATIVE(v)  ((v).type == VAL_NATIVE)
#define IS_OBJ(v)     ((v).type == VAL_OBJ)

// Helper macros extraction
#define AS_BOOL(v)    ((v).as.boolean)
#define AS_NUMBER(v)  ((v).as.number)
#define AS_TENSOR(v)  ((v).as.tensor_node->tensor) // Unpack through node
#define AS_TENSOR_NODE(v) ((v).as.tensor_node)     // Helper to get node
#define AS_NATIVE(v)  ((v).as.native)
#define AS_OBJ(v)     ((v).as.obj)

// Constructors
#define BOOL_VAL(value)   ((nc_value){VAL_BOOL, {.boolean = value}})
#define NIL_VAL           ((nc_value){VAL_NIL, {.number = 0}})
#define NUMBER_VAL(value) ((nc_value){VAL_NUMBER, {.number = value}})
#define TENSOR_VAL(node_ptr) ((nc_value){VAL_TENSOR, {.tensor_node = node_ptr}}) // Takes node
#define NATIVE_VAL(value) ((nc_value){VAL_NATIVE, {.native = value}})
#define OBJ_VAL(value)    ((nc_value){VAL_OBJ, {.obj = (nc_obj*)value}})

// Utilities
void nc_print_value(nc_value value);
bool nc_values_equal(nc_value a, nc_value b);

// Value Array (for constants pool)
typedef struct {
    int capacity;
    int count;
    nc_value* values;
} nc_value_array;

void nc_value_array_init(nc_value_array* array);
void nc_value_array_write(nc_value_array* array, nc_value value);
void nc_value_array_free(nc_value_array* array);

#endif // NOCTA_LANG_VALUE_H
