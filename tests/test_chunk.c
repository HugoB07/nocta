#include "minunit.h"
#include "nocta/lang/chunk.h"
#include "nocta/lang/opcode.h"

static char* test_chunk_write() {
    nc_chunk chunk;
    nc_chunk_init(&chunk);
    
    // Write constant
    int const_idx = nc_chunk_add_constant(&chunk, NUMBER_VAL(1.23));
    nc_chunk_write(&chunk, OP_CONSTANT, 1);
    nc_chunk_write(&chunk, const_idx, 1);
    
    // Write return
    nc_chunk_write(&chunk, OP_RETURN, 1);
    
    mu_assert("Count wrong", chunk.count == 3);
    mu_assert("Code[0] wrong", chunk.code[0] == OP_CONSTANT);
    mu_assert("Const count wrong", chunk.constants.count == 1);
    mu_assert("Const value wrong", AS_NUMBER(chunk.constants.values[0]) == 1.23);
    
    // Debug print
    printf("\n");
    nc_chunk_disassemble(&chunk, "test_chunk");
    
    nc_chunk_free(&chunk);
    return NULL;
}

char* test_chunk_suite() {
    mu_run_test(test_chunk_write);
    return NULL;
}
