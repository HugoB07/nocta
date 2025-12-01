#ifndef NOCTA_SERIALIZE_H
#define NOCTA_SERIALIZE_H

#include "nocta/core/tensor.h"
#include "nocta/core/error.h"
#include "nocta/nn/module.h"
#include <stdio.h>

// File format magix number: "NCTA" in ASCII
#define NC_FILE_MAGIC 0x4154434E

// Current file format version
#define NC_FILE_VERSION 1

// ============================================
// File header structure
// ============================================

typedef struct {
    uint32_t magic;     // NC_FILE_MAGIC
    uint32_t version;   // File format version
    uint32_t n_tensors; // Number of tensors in file
    uint32_t flags;     // Reserved for future use
    char description[64]; // Optional description
} nc_file_header;

// ============================================
// Tensor header (precedes each tensor's data)
// ============================================

typedef struct {
    char name[NC_MAX_NAME_LEN]; // Parameter name
    uint32_t dtype;             // Date type (nc_dtype)
    uint32_t ndim;              // Number of dimensions
    uint64_t shape[NC_MAX_DIMS]; // Shape array
    uint64_t data_size;          // Size of data in bytes
} nc_tensor_header;

// ============================================
// Tensor I/O
// ============================================

// Save a single tensor to file
nc_error nc_tensor_save(const nc_tensor* t, const char* path);

// Load a single tensor from file
nc_tensor* nc_tensor_load(const char* path);

// Save tensor to FILE stream
nc_error nc_tensor_save_fp(const nc_tensor* t, FILE* fp, const char* name);

// Load tensor from FILE stream
nc_tensor* nc_tensor_load_fp(FILE* fp, char* name_out);

// ============================================
// Module I/O
// ============================================

// Save module state dict (all parameters)
nc_error nc_module_save(nc_module* m, const char* path);

// Load module state dict
nc_error nc_module_load(nc_module* m, const char* path);

// Save with description
nc_error nc_module_save_desc(nc_module* m, const char* path, const char* description);

// ============================================
// State dict operations
// ============================================

// State dict: collection of named tensors
typedef struct {
    char** names;
    nc_tensor** tensors;
    size_t n_tensors;
    size_t capacity;
} nc_state_dict;

// Create empty state dict
nc_state_dict* nc_state_dict_create(void);

// Free state dict
void nc_state_dict_free(nc_state_dict* sd);

// Add tensor to state dict (takes ownership)
void nc_state_dict_add(nc_state_dict* sd, const char* name, nc_tensor* t);

// Get tensor by name (returns reference, not copy)
nc_tensor* nc_state_dict_get(nc_state_dict* sd, const char* name);

// Save state dict to file
nc_error nc_state_dict_save(nc_state_dict* sd, const char* path);

// Load state dict from file
nc_state_dict* nc_state_dict_load(const char* path);

// Extract state dict from module
nc_state_dict* nc_module_state_dict(nc_module* m);

// Load state dict into module
nc_error nc_module_load_state_dict(nc_module* m, nc_state_dict* sd);

// ============================================
// Checkpoint utilities
// ============================================

typedef struct {
    nc_state_dict* model_state;
    nc_state_dict* optimizer_state;
    size_t epoch;
    double loss;
} nc_checkpoint;

// Save training checkpoint
nc_error nc_checkpoint_save(nc_checkpoint* ckpt, const char* path);

// Load training checkpoint
nc_checkpoint* nc_checkpoint_load(const char* path);

// Free checkpoint
void nc_checkpoint_free(nc_checkpoint* ckpt);

// ============================================
// Utility
// ============================================

// Print file info without loading
void nc_file_info(const char* path);

// Verify file integrity
bool nc_file_verify(const char* path);

#endif // NOCTA_SERIALIZE_H