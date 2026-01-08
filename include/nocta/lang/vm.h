#ifndef NOCTA_LANG_VM_H
#define NOCTA_LANG_VM_H

#include "nocta/lang/object.h"
#include "nocta/lang/value.h"

#define FRAMES_MAX 64
#define STACK_MAX (FRAMES_MAX * 256)

typedef struct {
    nc_function* function; // The function being executed
    uint8_t* ip;           // Current instruction pointer in function's chunk
    nc_value* slots;       // Points to first stack slot for this frame (locals)
} nc_call_frame;

struct nc_vm {
    // Stack
    nc_value stack[STACK_MAX];
    nc_value* stack_top;
    
    // Call Frames
    nc_call_frame frames[FRAMES_MAX];
    int frame_count;
    
    // Globals
    struct {
        int capacity;
        int count;
        const char** names;
        nc_value* values;
    } globals;
    
    struct {
        const char* names[64];
        nc_native_fn funcs[64];
        int count;
    } natives;

    // Tensor tracking for memory management
    struct nc_tensor_node* tensors;
};

typedef enum {
    INTERPRET_OK,
    INTERPRET_COMPILE_ERROR,
    INTERPRET_RUNTIME_ERROR
} nc_interpret_result;

void nc_vm_init(nc_vm* vm);
void nc_vm_free(nc_vm* vm);
nc_interpret_result nc_vm_interpret(nc_vm* vm, nc_chunk* chunk);

// Stack operations
void nc_vm_push(nc_vm* vm, nc_value value);
nc_value nc_vm_pop(nc_vm* vm);

// Natives
void nc_vm_define_native(nc_vm* vm, const char* name, nc_native_fn function);

#endif // NOCTA_LANG_VM_H
