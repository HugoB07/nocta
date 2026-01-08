#ifndef NOCTA_LANG_COMPILER_H
#define NOCTA_LANG_COMPILER_H

#include "ast.h"
#include "chunk.h"

// Compile AST into Bytecode Chunk
// Returns true if compilation succeeded, false otherwise.
bool nc_compile(nc_ast_program* program, nc_chunk* chunk);

#endif // NOCTA_LANG_COMPILER_H
