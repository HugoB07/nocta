#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "nocta/lang/object.h"
#include "nocta/lang/value.h"
#include "nocta/lang/chunk.h"

static nc_obj* objects = NULL;

void nc_free_objects() {
    nc_obj* object = objects;
    while (object != NULL) {
        nc_obj* next = object->next;
        // Free object based on type
        switch (object->type) {
            case OBJ_STRING: {
                nc_string* string = (nc_string*)object;
                free(string->chars);
                free(string);
                break;
            }
            case OBJ_LIST: {
                nc_list* list = (nc_list*)object;
                free(list->items); 
                free(list);
                break;
            }
            case OBJ_RANGE: {
                 free(object);
                 break;
            }
            // Add other types...
            default:
                free(object);
                break;
        }
        object = next;
    }
    objects = NULL;
}

// Sweep unmarked objects
void nc_sweep_objects() {
    nc_obj** object = &objects;
    while (*object != NULL) {
        if ((*object)->is_marked) {
            (*object)->is_marked = false; // Unmark for next cycle
            object = &(*object)->next;
        } else {
            nc_obj* unreached = *object;
            *object = unreached->next;
            
            // Free unreached
            switch (unreached->type) {
                case OBJ_STRING: {
                    nc_string* string = (nc_string*)unreached;
                    free(string->chars);
                    free(string);
                    break;
                }
                case OBJ_LIST: {
                    nc_list* list = (nc_list*)unreached;
                    free(list->items); 
                    free(list);
                    break;
                }
                case OBJ_RANGE: {
                     free(unreached);
                     break;
                }
                default:
                    free(unreached);
                    break; 
            }
        }
    }
}

static nc_obj* allocate_obj(size_t size, nc_obj_type type) {
    nc_obj* object = (nc_obj*)malloc(size);
    if (object == NULL) {
        fprintf(stderr, "Out of memory allocating object.\n");
        exit(1);
    }
    object->type = type;
    object->is_marked = false;
    object->next = objects;
    objects = object;
    return object;
}

static nc_string* allocate_string(char* chars, int length, uint32_t hash) {
    nc_string* string = (nc_string*)allocate_obj(sizeof(nc_string), OBJ_STRING);
    string->length = length;
    string->chars = chars;
    string->hash = hash;
    return string;
}

static uint32_t hash_string(const char* key, int length) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < length; i++) {
        hash ^= (uint8_t)key[i];
        hash *= 16777619;
    }
    return hash;
}

nc_string* nc_allocate_string(const char* chars, int length) {
    char* heap_chars = malloc(length + 1);
    memcpy(heap_chars, chars, length);
    heap_chars[length] = '\0';
    return allocate_string(heap_chars, length, hash_string(chars, length));
}

nc_string* nc_take_string(char* chars, int length) {
    return allocate_string(chars, length, hash_string(chars, length));
}

nc_function* nc_new_function() {
    nc_function* function = (nc_function*)allocate_obj(sizeof(nc_function), OBJ_FUNCTION);
    function->arity = 0;
    function->upvalue_count = 0;
    function->name = NULL;
    nc_chunk_init(&function->chunk);
    return function;
}

nc_list* nc_new_list() {
    nc_list* list = (nc_list*)allocate_obj(sizeof(nc_list), OBJ_LIST);
    list->count = 0;
    list->capacity = 0;
    list->items = NULL;
    return list;
}

nc_range* nc_new_range(int start, int end) {
    nc_range* range = (nc_range*)allocate_obj(sizeof(nc_range), OBJ_RANGE);
    range->start = start;
    range->end = end;
    range->current = start;
    return range;
}

void nc_list_append(nc_list* list, nc_value value) {
    if (list->capacity < list->count + 1) {
        int old_capacity = list->capacity;
        list->capacity = (old_capacity < 8) ? 8 : old_capacity * 2;
        list->items = (nc_value*)realloc(list->items, sizeof(nc_value) * list->capacity);
    }
    list->items[list->count] = value;
    list->count++;
}

nc_class* nc_new_class(nc_string* name) {
    nc_class* klass = (nc_class*)allocate_obj(sizeof(nc_class), OBJ_CLASS);
    klass->name = name;
    klass->methods = NULL;
    klass->method_count = 0;
    klass->method_capacity = 0;
    return klass;
}

void nc_class_add_method(nc_class* klass, nc_string* name, nc_function* method) {
    if (klass->method_count + 1 > klass->method_capacity) {
        int capacity = klass->method_capacity < 8 ? 8 : klass->method_capacity * 2;
        klass->methods = realloc(klass->methods, sizeof(*klass->methods) * capacity);
        klass->method_capacity = capacity;
    }
    
    klass->methods[klass->method_count].key = name;
    klass->methods[klass->method_count].value = (nc_obj*)method;
    klass->method_count++;
}

nc_bound_method* nc_new_bound_method(nc_value receiver, nc_function* method) {
    nc_bound_method* bound = (nc_bound_method*)allocate_obj(sizeof(nc_bound_method), OBJ_BOUND_METHOD);
    bound->receiver = receiver;
    bound->method = method;
    return bound;
}

nc_instance* nc_new_instance(nc_class* klass) {
    nc_instance* instance = (nc_instance*)allocate_obj(sizeof(nc_instance), OBJ_INSTANCE);
    instance->klass = klass;
    instance->fields = NULL;
    instance->field_count = 0;
    instance->field_capacity = 0;
    return instance;
}

void nc_instance_set_field(nc_instance* instance, nc_string* name, nc_value value) {
    // Linear scan for now
    for (int i = 0; i < instance->field_count; i++) {
        // String interning assumption: Pointers match if strings match (requires interning)
        // OR deep comparison. For now, assuming distinct pointers but check chars for safety
        // if interning isn't fully robust yet.
        // Actually, if we rely on string values, we should compare hashes or chars?
        // Let's assume naive char usage if no hash. But we have hash.
        
        // Wait, nc_string has hash.
        if (instance->fields[i].key->hash == name->hash &&
            instance->fields[i].key->length == name->length &&
            memcmp(instance->fields[i].key->chars, name->chars, name->length) == 0) {
            
            instance->fields[i].value = value;
            return;
        }
    }
    
    // Add new field
    if (instance->field_count + 1 > instance->field_capacity) {
        int capacity = instance->field_capacity < 8 ? 8 : instance->field_capacity * 2;
        instance->fields = realloc(instance->fields, sizeof(*instance->fields) * capacity);
        instance->field_capacity = capacity;
    }
    
    instance->fields[instance->field_count].key = name;
    instance->fields[instance->field_count].value = value;
    instance->field_count++;
}

bool nc_instance_get_field(nc_instance* instance, nc_string* name, nc_value* value) {
    for (int i = 0; i < instance->field_count; i++) {
        if (instance->fields[i].key->hash == name->hash &&
            instance->fields[i].key->length == name->length &&
            memcmp(instance->fields[i].key->chars, name->chars, name->length) == 0) {
            
            *value = instance->fields[i].value;
            return true;
        }
    }
    return false;
}

void nc_free_obj(nc_obj* obj) {
    switch (obj->type) {
        case OBJ_STRING: {
            nc_string* string = (nc_string*)obj;
            free(string->chars);
            free(string);
            break;
        }
        case OBJ_FUNCTION: {
            nc_function* function = (nc_function*)obj;
            nc_chunk_free(&function->chunk);
            free(function); // Name is managed as separate string obj
            break;
        }
        case OBJ_RANGE: {
            // Nothing extra to free, just the struct
            free(obj);
            break;
        }
        case OBJ_SCRIPT: {
            nc_function* function = (nc_function*)obj;
            nc_chunk_free(&function->chunk);
            free(function);
            break;
        }
        case OBJ_NATIVE_FUNCTION:
            free(obj);
            break;
        case OBJ_CLASS: {
            nc_class* klass = (nc_class*)obj;
            if (klass->methods) free(klass->methods);
            free(obj);
            break;
        }
        case OBJ_INSTANCE: {
            nc_instance* instance = (nc_instance*)obj;
            if (instance->fields) free(instance->fields);
            free(obj);
            break;
        }
        case OBJ_BOUND_METHOD: {
            free(obj);
            break;
        }
    }
}
