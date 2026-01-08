#ifndef NOCTA_LANG_OBJECT_H
#define NOCTA_LANG_OBJECT_H

#include "nocta/lang/value.h"
#include "nocta/lang/chunk.h"

typedef enum {
    OBJ_STRING,
    OBJ_FUNCTION,
    OBJ_NATIVE_FUNCTION,
    OBJ_SCRIPT,
    OBJ_RANGE,
    OBJ_CLASS,
    OBJ_INSTANCE,
    OBJ_BOUND_METHOD,
    OBJ_LIST
} nc_obj_type;

struct nc_obj {
    nc_obj_type type;
    bool is_marked;
    struct nc_obj* next; // For GC
};

typedef struct {
    nc_obj obj;
    int length;
    char* chars;
    uint32_t hash; // For string interning/maps
} nc_string;

typedef struct {
    nc_obj obj;
    int arity;
    int upvalue_count; // Future proofing
    nc_chunk chunk;
    nc_string* name;
} nc_function;

typedef struct {
    nc_obj obj;
    int start;
    int end;
    int current;
} nc_range;

typedef struct {
    nc_obj obj;
    nc_string* name;
    // Methods hash map
    struct {
        nc_string* key;
        nc_obj* value; // nc_function* (actually nc_closure* later? For now nc_function*)
    } *methods;
    int method_count;
    int method_capacity;
} nc_class;

typedef struct {
    nc_obj obj;
    nc_class* klass;
    // Simple hash map (linear probe or array for MVP)
    struct {
        nc_string* key;
        nc_value value;
    } *fields;
    int field_count;
    int field_capacity;
} nc_instance;

typedef struct {
    nc_obj obj;
    nc_value receiver; // Instance
    nc_function* method; // The function
} nc_bound_method;

typedef struct {
    nc_obj obj;
    int count;
    int capacity;
    nc_value* items;
} nc_list; 

// Macros for type checking
#define OBJ_TYPE(value)   (AS_OBJ(value)->type)
#define IS_STRING(value)  (is_obj_type(value, OBJ_STRING))
#define IS_FUNCTION(value) (is_obj_type(value, OBJ_FUNCTION))
#define IS_LIST(value)     (is_obj_type(value, OBJ_LIST))
#define IS_RANGE(value)    (is_obj_type(value, OBJ_RANGE))
#define IS_CLASS(value)    (is_obj_type(value, OBJ_CLASS))
#define IS_INSTANCE(value) (is_obj_type(value, OBJ_INSTANCE))
#define IS_BOUND_METHOD(value) (is_obj_type(value, OBJ_BOUND_METHOD))

// Macros for casting
#define AS_STRING(value)   ((nc_string*)AS_OBJ(value))
#define AS_CSTRING(value)  (((nc_string*)AS_OBJ(value))->chars)
#define AS_FUNCTION(value) ((nc_function*)AS_OBJ(value))
#define AS_LIST(value)     ((nc_list*)AS_OBJ(value))
#define AS_RANGE(value)    ((nc_range*)AS_OBJ(value))
#define AS_CLASS(value)    ((nc_class*)AS_OBJ(value))
#define AS_INSTANCE(value) ((nc_instance*)AS_OBJ(value))
#define AS_BOUND_METHOD(value) ((nc_bound_method*)AS_OBJ(value))

static inline bool is_obj_type(nc_value value, nc_obj_type type) {
    return IS_OBJ(value) && AS_OBJ(value)->type == type;
}

// Allocation functions
nc_string* nc_allocate_string(const char* chars, int length);
nc_string* nc_take_string(char* chars, int length);
nc_function* nc_new_function();
nc_range* nc_new_range(int start, int end);
nc_class* nc_new_class(nc_string* name);
void nc_class_add_method(nc_class* klass, nc_string* name, nc_function* method);
nc_instance* nc_new_instance(nc_class* klass);
nc_bound_method* nc_new_bound_method(nc_value receiver, nc_function* method);
// Memory management
void nc_free_objects();
void nc_sweep_objects();

nc_list* nc_new_list();
void nc_list_append(nc_list* list, nc_value value);
void nc_instance_set_field(nc_instance* instance, nc_string* name, nc_value value);
bool nc_instance_get_field(nc_instance* instance, nc_string* name, nc_value* value);
void nc_free_obj(nc_obj* obj);

#endif // NOCTA_LANG_OBJECT_H
