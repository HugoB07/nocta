#ifndef NOCTA_LANG_CHUNK_H
#define NOCTA_LANG_CHUNK_H

#include "nocta/lang/value.h"
#include "nocta/lang/opcode.h"

// A chunk of bytecode
typedef struct {
    int count;
    int capacity;
    uint8_t* code;
    int* lines;           // Line number for each byte
    nc_value_array constants; // Pool of constants (numbers, strings)
} nc_chunk;

void nc_chunk_init(nc_chunk* chunk);
void nc_chunk_free(nc_chunk* chunk);
void nc_chunk_write(nc_chunk* chunk, uint8_t byte, int line);
int nc_chunk_add_constant(nc_chunk* chunk, nc_value value);

// Debug
void nc_chunk_disassemble(nc_chunk* chunk, const char* name);

#endif // NOCTA_LANG_CHUNK_H
