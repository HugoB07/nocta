#include "nocta/lang/value.h"
#include "nocta/lang/object.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// ==========================================
// Value Array
// ==========================================

void nc_value_array_init(nc_value_array* array) {
    array->values = NULL;
    array->capacity = 0;
    array->count = 0;
}

void nc_value_array_write(nc_value_array* array, nc_value value) {
    if (array->capacity < array->count + 1) {
        int old_capacity = array->capacity;
        array->capacity = (old_capacity < 8) ? 8 : old_capacity * 2;
        array->values = (nc_value*)realloc(array->values, sizeof(nc_value) * array->capacity);
    }
    array->values[array->count] = value;
    array->count++;
}

void nc_value_array_free(nc_value_array* array) {
    free(array->values);
    nc_value_array_init(array);
}

void nc_print_value(nc_value value) {
    switch (value.type) {
        case VAL_BOOL:   printf(AS_BOOL(value) ? "true" : "false"); break;
        case VAL_NIL:    printf("nil"); break;
        case VAL_NUMBER: printf("%g", AS_NUMBER(value)); break;
        case VAL_TENSOR: {
            nc_tensor* t = AS_TENSOR(value);
            if (t) {
                printf("<tensor [");
                for (int i = 0; i < t->ndim; i++) {
                    printf("%zu%s", t->shape[i], (i < t->ndim - 1) ? "x" : "");
                }
                printf("]>");
            } else {
                printf("<tensor (null)>");
            }
            break;
        }
        case VAL_NATIVE: printf("<native fn>"); break;
        case VAL_OBJ: {
            switch (OBJ_TYPE(value)) {
                case OBJ_STRING: printf("%s", AS_CSTRING(value)); break;
                case OBJ_FUNCTION: {
                    if (AS_FUNCTION(value)->name == NULL) {
                        printf("<script>");
                    } else {
                        printf("<fn %s>", AS_FUNCTION(value)->name->chars);
                    }
                    break;
                }
                case OBJ_CLASS: {
                    printf("<class %s>", AS_CLASS(value)->name->chars);
                    break;
                }
                case OBJ_INSTANCE: {
                    printf("<instance %s>", AS_INSTANCE(value)->klass->name->chars);
                    break;
                }
                case OBJ_BOUND_METHOD: {
                    printf("<method %s>", AS_BOUND_METHOD(value)->method->name->chars);
                    break;
                }
                case OBJ_LIST: {
                    nc_list* list = AS_LIST(value);
                    printf("[");
                    for (int i = 0; i < list->count; i++) {
                        nc_print_value(list->items[i]);
                        if (i < list->count - 1) printf(", ");
                    }
                    printf("]");
                    break;
                }
            }
            break;
        }
    }
}

bool nc_values_equal(nc_value a, nc_value b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case VAL_BOOL:   return AS_BOOL(a) == AS_BOOL(b);
        case VAL_NIL:    return true;
        case VAL_NUMBER: return AS_NUMBER(a) == AS_NUMBER(b);
        case VAL_TENSOR: return AS_TENSOR(a) == AS_TENSOR(b); 
        case VAL_NATIVE: return AS_NATIVE(a) == AS_NATIVE(b);
        case VAL_OBJ:    return a.as.obj == b.as.obj; // Pointer equality for now
        default:         return false;
    }
}
