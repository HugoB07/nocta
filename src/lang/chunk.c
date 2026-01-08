#include "nocta/lang/chunk.h"
#include <stdlib.h>
#include <stdio.h>

// ==========================================
// Chunk
// ==========================================

void nc_chunk_init(nc_chunk* chunk) {
    chunk->count = 0;
    chunk->capacity = 0;
    chunk->code = NULL;
    chunk->lines = NULL;
    nc_value_array_init(&chunk->constants);
}

void nc_chunk_free(nc_chunk* chunk) {
    free(chunk->code);
    free(chunk->lines);
    nc_value_array_free(&chunk->constants);
    nc_chunk_init(chunk);
}

void nc_chunk_write(nc_chunk* chunk, uint8_t byte, int line) {
    if (chunk->capacity < chunk->count + 1) {
        int old_capacity = chunk->capacity;
        chunk->capacity = (old_capacity < 8) ? 8 : old_capacity * 2;
        chunk->code = (uint8_t*)realloc(chunk->code, sizeof(uint8_t) * chunk->capacity);
        chunk->lines = (int*)realloc(chunk->lines, sizeof(int) * chunk->capacity);
    }
    
    chunk->code[chunk->count] = byte;
    chunk->lines[chunk->count] = line;
    chunk->count++;
}

int nc_chunk_add_constant(nc_chunk* chunk, nc_value value) {
    nc_value_array_write(&chunk->constants, value);
    return chunk->constants.count - 1;
}

// ==========================================
// Disassembler (Debug)
// ==========================================

static int simple_instruction(const char* name, int offset) {
    printf("%s\n", name);
    return offset + 1;
}

static int constant_instruction(const char* name, nc_chunk* chunk, int offset) {
    uint8_t constant = chunk->code[offset + 1];
    printf("%-16s %4d '", name, constant);
    nc_print_value(chunk->constants.values[constant]);
    printf("'\n");
    return offset + 2;
}

static int byte_instruction(const char* name, nc_chunk* chunk, int offset) {
    uint8_t slot = chunk->code[offset + 1];
    printf("%-16s %4d\n", name, slot);
    return offset + 2;
}

static int jump_instruction(const char* name, int sign, nc_chunk* chunk, int offset) {
    uint16_t jump = (uint16_t)(chunk->code[offset + 1] << 8);
    jump |= chunk->code[offset + 2];
    printf("%-16s %4d -> %d\n", name, offset, offset + 3 + sign * jump);
    return offset + 3;
}

void nc_chunk_disassemble(nc_chunk* chunk, const char* name) {
    printf("== %s ==\n", name);
    
    for (int offset = 0; offset < chunk->count;) {
        printf("%04d ", offset);
        if (offset > 0 && chunk->lines[offset] == chunk->lines[offset - 1]) {
            printf("   | ");
        } else {
            printf("%4d ", chunk->lines[offset]);
        }
        
        uint8_t instruction = chunk->code[offset];
        switch (instruction) {
            case OP_RETURN:     offset = simple_instruction("OP_RETURN", offset); break;
            case OP_CONSTANT:   offset = constant_instruction("OP_CONSTANT", chunk, offset); break;
            case OP_NIL:        offset = simple_instruction("OP_NIL", offset); break;
            case OP_TRUE:       offset = simple_instruction("OP_TRUE", offset); break;
            case OP_FALSE:      offset = simple_instruction("OP_FALSE", offset); break;
            case OP_POP:        offset = simple_instruction("OP_POP", offset); break;
            case OP_GET_LOCAL:  offset = byte_instruction("OP_GET_LOCAL", chunk, offset); break;
            case OP_SET_LOCAL:  offset = byte_instruction("OP_SET_LOCAL", chunk, offset); break;
            case OP_DEFINE_GLOBAL: offset = constant_instruction("OP_DEFINE_GLOBAL", chunk, offset); break;
            case OP_GET_GLOBAL: offset = constant_instruction("OP_GET_GLOBAL", chunk, offset); break;
            case OP_SET_GLOBAL: offset = constant_instruction("OP_SET_GLOBAL", chunk, offset); break;
            case OP_EQUAL:      offset = simple_instruction("OP_EQUAL", offset); break;
            case OP_GREATER:    offset = simple_instruction("OP_GREATER", offset); break;
            case OP_LESS:       offset = simple_instruction("OP_LESS", offset); break;
            case OP_ADD:        offset = simple_instruction("OP_ADD", offset); break;
            case OP_SUB:        offset = simple_instruction("OP_SUB", offset); break;
            case OP_MUL:        offset = simple_instruction("OP_MUL", offset); break;
            case OP_DIV:        offset = simple_instruction("OP_DIV", offset); break;
            case OP_MOD:        offset = simple_instruction("OP_MOD", offset); break;
            case OP_NOT:        offset = simple_instruction("OP_NOT", offset); break;
            case OP_NEGATE:     offset = simple_instruction("OP_NEGATE", offset); break;
            case OP_PRINT:      offset = simple_instruction("OP_PRINT", offset); break;
            case OP_JUMP:       offset = jump_instruction("OP_JUMP", 1, chunk, offset); break;

            case OP_CALL:       offset = byte_instruction("OP_CALL", chunk, offset); break;
            case OP_FOR_ITER:   offset = jump_instruction("OP_FOR_ITER", 1, chunk, offset); break;
            
            case OP_CLASS: offset = constant_instruction("OP_CLASS", chunk, offset); break;
            case OP_GET_PROPERTY: offset = constant_instruction("OP_GET_PROPERTY", chunk, offset); break;
            case OP_SET_PROPERTY: offset = constant_instruction("OP_SET_PROPERTY", chunk, offset); break;
            case OP_METHOD: offset = constant_instruction("OP_METHOD", chunk, offset); break;
            
            default:
                printf("Unknown opcode %d\n", instruction);
                offset = offset + 1;
        }
    }
}
