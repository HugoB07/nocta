#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nocta/lang/vm.h"
#include "nocta/lang/compiler.h"
#include "nocta/lang/lexer.h"
#include "nocta/lang/parser.h"
#include "nocta/lang/chunk.h"
#include "nocta/nocta.h" 

static char* read_file(const char* path) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Could not open file \"%s\".\n", path);
        exit(74);
    }

    fseek(file, 0L, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);

    char* buffer = (char*)malloc(fileSize + 1);
    if (buffer == NULL) {
        fprintf(stderr, "Not enough memory to read \"%s\".\n", path);
        exit(74);
    }

    size_t bytesRead = fread(buffer, sizeof(char), fileSize, file);
    if (bytesRead < fileSize) {
        fprintf(stderr, "Could not read file \"%s\".\n", path);
        exit(74);
    }

    buffer[bytesRead] = '\0';

    fclose(file);
    return buffer;
}

static bool interpret_source(nc_vm* vm, const char* source) {
    nc_lexer lexer;
    nc_lexer_init(&lexer, source);
    
    nc_parser parser;
    nc_parser_init(&parser, &lexer);
    
    nc_ast_program* prog = nc_parser_parse(&parser);
    if (!prog || parser.had_error) {
        if (prog) nc_ast_free((nc_ast_node*)prog);
        return false;
    }
    
    nc_chunk chunk;
    nc_chunk_init(&chunk);
    
    if (!nc_compile(prog, &chunk)) {
        nc_chunk_free(&chunk);
        nc_ast_free((nc_ast_node*)prog);
        return false;
    }
    
    nc_interpret_result result = nc_vm_interpret(vm, &chunk);
    
    nc_chunk_free(&chunk);
    nc_ast_free((nc_ast_node*)prog);

    if (result == INTERPRET_COMPILE_ERROR) return false;
    if (result == INTERPRET_RUNTIME_ERROR) return false;
    return true;
}

static void run_file(const char* path) {
    char* source = read_file(path);
    
    nc_vm vm;
    nc_vm_init(&vm);
    
    bool success = interpret_source(&vm, source);
    
    nc_vm_free(&vm);
    free(source);
    
    if (!success) exit(65); // Simplified exit code
}

static void run_repl() {
    printf("Nocta v%s\n", NOCTA_VERSION);
    printf("Type 'exit' to quit.\n");
    
    nc_vm vm;
    nc_vm_init(&vm);
    
    char line[1024];
    for (;;) {
        printf("> ");
        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n");
            break;
        }
        
        // Remove trailing newline
        line[strcspn(line, "\n")] = 0;

        if (strncmp(line, "exit", 4) == 0 && (line[4] == '\0' || line[4] == ' ')) break;
        
        interpret_source(&vm, line);
    }
    
    nc_vm_free(&vm);
}

int main(int argc, char* argv[]) {
    nc_init();
    
    if (argc == 1) {
        run_repl();
    } else if (argc == 2) {
        run_file(argv[1]);
    } else {
        fprintf(stderr, "Usage: nocta_cli [path]\n");
        exit(64);
    }
    
    nc_cleanup();
    return 0;
}
