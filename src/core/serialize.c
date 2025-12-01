#include "nocta/core/serialize.h"
#include "nocta/core/memory.h"
#include <string.h>
#include <stdlib.h>

// ============================================
// Helper functions
// ============================================

static bool write_header(FILE* fp, const nc_file_header* header) {
    return fwrite(header, sizeof(nc_file_header), 1, fp) == 1;
}

static bool read_header(FILE* fp, nc_file_header* header) {
    if(fread(header, sizeof(nc_file_header), 1, fp) != 1) return false;
    if(header->magic != NC_FILE_MAGIC) return false;
    return true;
}

static bool write_tensor_header(FILE* fp, const nc_tensor_header* th) {
    return fwrite(th, sizeof(nc_tensor_header), 1, fp) == 1;
}

static bool read_tensor_header(FILE* fp, nc_tensor_header* th) {
    return fread(th, sizeof(nc_tensor_header), 1, fp) == 1;
}

// ============================================
// Tensor I/O
// ============================================

nc_error nc_tensor_save_fp(const nc_tensor* t, FILE* fp, const char* name) {
    if(!t || !fp) return NC_ERR_NULL_PTR;

    nc_tensor_header th = {0};
    if(name) {
        strncpy(th.name, name, NC_MAX_NAME_LEN - 1);
    }
    th.dtype = (uint32_t)t->dtype;
    th.ndim = (uint32_t)t->ndim;
    for(size_t  i = 0; i < t->ndim; i++) {
        th.shape[i] = t->shape[i];
    }
    th.data_size = t->numel * nc_dtype_sizeof(t->dtype);

    if(!write_tensor_header(fp, &th)) {
        return NC_ERR_IO;
    }

    // Write tensor data
    // If non-contiguous, need to copy to contiguous buffer
    if(t->is_contiguous) {
        if(fwrite(nc_tensor_data(t), 1, th.data_size, fp) != th.data_size) {
            return NC_ERR_IO;
        }
    }
    else {
        // Make contiguous copy
        nc_tensor* contig = nc_tensor_contiguous((nc_tensor*)t);
        if(!contig) return NC_ERR_ALLOC;

        size_t written = fwrite(nc_tensor_data(contig), 1, th.data_size, fp);
        nc_tensor_free(contig);

        if(written != th.data_size) return NC_ERR_IO;
    }

    return NC_OK;
}

nc_tensor* nc_tensor_load_fp(FILE* fp, char* name_out) {
    if(!fp) return NULL;

    nc_tensor_header th;
    if(!read_tensor_header(fp, &th)) {
        NC_SET_ERROR(NC_ERR_IO, "Failed to read tensor header");
        return NULL;
    }

    if(name_out) {
        strncpy(name_out, th.name, NC_MAX_NAME_LEN - 1);
    }

    // Reconstruct shape
    size_t shape[NC_MAX_DIMS];
    for(size_t i = 0; i < th.ndim; i++) {
        shape[i] = (size_t)th.shape[i];
    }

    nc_tensor* t = nc_tensor_empty(shape, th.ndim, (nc_dtype)th.dtype);
    if(!t) return NULL;

    // Read data directly into tensor storage
    if(fread(nc_tensor_data(t), 1, th.data_size, fp) != th.data_size) {
        nc_tensor_free(t);
        NC_SET_ERROR(NC_ERR_IO, "Failed to read tensor data");
        return NULL;
    }

    return t;
}

nc_error nc_tensor_save(const nc_tensor* t, const char* path) {
    if(!t || !path) return NC_ERR_NULL_PTR;

    FILE* fp = fopen(path, "wb");
    if(!fp) {
        NC_SET_ERROR(NC_ERR_IO, "Cannot open file: %s", path);
        return NC_ERR_IO;
    }

    // Write file header
    nc_file_header header = {
        .magic = NC_FILE_MAGIC,
        .version = NC_FILE_VERSION,
        .n_tensors = 1,
        .flags = 0
    };
    strncpy(header.description, "Single tensor", sizeof(header.description) - 1);

    if(!write_header(fp, &header)) {
        fclose(fp);
        return NC_ERR_IO;
    }

    nc_error err = nc_tensor_save_fp(t, fp, "tensor");
    fclose(fp);
    return err;
}

nc_tensor* nc_tensor_load(const char* path) {
    if(!path) return NULL;

    FILE* fp = fopen(path, "rb");
    if(!fp) {
        NC_SET_ERROR(NC_ERR_IO, "Cannot open file: %s", path);
        return NULL;
    }

    nc_file_header header;
    if(!read_header(fp, &header)) {
        fclose(fp);
        NC_SET_ERROR(NC_ERR_IO, "Invalid file format");
        return NULL;
    }

    nc_tensor* t = nc_tensor_load_fp(fp, NULL);
    fclose(fp);
    return t;
}

// ============================================
// State dict
// ============================================

nc_state_dict* nc_state_dict_create(void) {
    nc_state_dict* sd = nc_calloc(1, sizeof(nc_state_dict));
    if(!sd) return NULL;

    sd->capacity = 16;
    sd->names = nc_calloc(sd->capacity, sizeof(char*));
    sd->tensors = nc_calloc(sd->capacity, sizeof(nc_tensor*));

    if(!sd->names || !sd->tensors) {
        nc_free(sd->names);
        nc_free(sd->tensors);
        nc_free(sd);
        return NULL;
    }

    return sd;
}

void nc_state_dict_free(nc_state_dict* sd) {
    if(!sd) return;

    for(size_t i = 0; i < sd->n_tensors; i++) {
        nc_free(sd->names[i]);
        nc_tensor_free(sd->tensors[i]);
    }

    nc_free(sd->names);
    nc_free(sd->tensors);
    nc_free(sd);
}

void nc_state_dict_add(nc_state_dict* sd, const char* name, nc_tensor* t) {
    if(!sd || !name || !t) return;

    // Grow if needed
    if(sd->n_tensors >= sd->capacity) {
        size_t new_cap = sd->capacity * 2;
        char** new_names = nc_realloc(sd->names, sd->capacity * sizeof(char*), new_cap * sizeof(char*));
        nc_tensor** new_tensors = nc_realloc(sd->tensors, sd->capacity * sizeof(nc_tensor*), new_cap * sizeof(nc_tensor*));

        if(!new_names || !new_tensors) return;

        sd->names = new_names;
        sd->tensors = new_tensors;
        sd->capacity = new_cap;
    }

    // Add entry
    sd->names[sd->n_tensors] = nc_alloc(strlen(name) + 1);
    if(sd->names[sd->n_tensors]) {
        strcpy(sd->names[sd->n_tensors], name);
    }
    sd->tensors[sd->n_tensors] = t;
    sd->n_tensors++;
}

nc_tensor* nc_state_dict_get(nc_state_dict* sd, const char* name) {
    if(!sd || !name) return NULL;

    for(size_t i = 0; i < sd->n_tensors; i++) {
        if(strcmp(sd->names[i], name) == 0) {
            return sd->tensors[i];
        }
    }
    return NULL;
}

nc_error nc_state_dict_save(nc_state_dict* sd, const char* path) {
    if(!sd || !path) return NC_ERR_NULL_PTR;

    FILE* fp = fopen(path, "wb");
    if(!fp) {
        NC_SET_ERROR(NC_ERR_IO, "Cannot open file: %s", path);
        return NC_ERR_IO;
    }

    // Write header
    nc_file_header header = {
        .magic = NC_FILE_MAGIC,
        .version = NC_FILE_VERSION,
        .n_tensors = (uint32_t)sd->n_tensors,
        .flags = 0
    };
    strncpy(header.description, "State dict", sizeof(header.description) - 1);

    if(!write_header(fp, &header)) {
        fclose(fp);
        return NC_ERR_IO;
    }

    // Write each tensor
    for(size_t i = 0; i < sd->n_tensors; i++) {
        nc_error err = nc_tensor_save_fp(sd->tensors[i], fp, sd->names[i]);
        if(err != NC_OK) {
            fclose(fp);
            return err;
        }
    }

    fclose(fp);
    return NC_OK;
}

nc_state_dict* nc_state_dict_load(const char* path) {
    if(!path) return NULL;

    FILE* fp = fopen(path, "rb");
    if(!fp) {
        NC_SET_ERROR(NC_ERR_IO, "Cannot open file: %s", path);
        return NULL;
    }

    nc_file_header header;
    if(!read_header(fp, &header)) {
        fclose(fp);
        NC_SET_ERROR(NC_ERR_IO, "Invalid file format");
        return NULL;
    }

    nc_state_dict* sd = nc_state_dict_create();
    if(!sd) {
        fclose(fp);
        return NULL;
    }

    // Load each tensor
    for(uint32_t i = 0; i < header.n_tensors; i++) {
        char name[NC_MAX_NAME_LEN] = {0};
        nc_tensor* t = nc_tensor_load_fp(fp, name);

        if(!t) {
            nc_state_dict_free(sd);
            fclose(fp);
            return NULL;
        }

        nc_state_dict_add(sd, name, t);
    }

    fclose(fp);
    return sd;
}

// ============================================
// Module state dict operations
// ============================================

// Callback context for extracting state dict
typedef struct {
    nc_state_dict* sd;
} extract_ctx;

static void extract_param_cb(const char* name, nc_tensor* param, void* ctx) {
    extract_ctx* ec = ctx;
    // Clone tensor for state dict
    nc_tensor* clone = nc_tensor_clone(param);
    if(clone) {
        nc_state_dict_add(ec->sd, name, clone);
    }
}

nc_state_dict* nc_module_state_dict(nc_module* m) {
    if(!m) return NULL;

    nc_state_dict* sd = nc_state_dict_create();
    if(!sd) return NULL;

    extract_ctx ctx = { .sd = sd };
    nc_module_parameters(m, extract_param_cb, &ctx);

    return sd;
} 

nc_error nc_module_load_state_dict(nc_module* m, nc_state_dict* sd) {
    if(!m || !sd) return NC_ERR_NULL_PTR;

    // For each tensor in state dict, find matching param and copy
    for(size_t i = 0; i < sd->n_tensors; i++) {
        nc_tensor* param = nc_module_get_param(m, sd->names[i]);
        if(!param) {
            // Try to find in submodules (handle "layer.weight" style names)
            // For now, skip if not found directly
            NC_WARN("Parameter not found: %s", sd->names[i]);
            continue;
        }

        nc_tensor* loaded = sd->tensors[i];

        // Shape check
        if(!nc_tensor_shape_eq(param, loaded)) {
            NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "Shape mismatch for %s", sd->names[i]);
            return NC_ERR_SHAPE_MISMATCH;
        }

        // Copy data
        nc_tensor_copy_(param, loaded);
    }

    return NC_OK;
}

// ============================================
// Module I/O (convenience wrappers)
// ============================================

nc_error nc_module_save(nc_module* m, const char* path) {
    if(!m || !path) return NC_ERR_NULL_PTR;

    nc_state_dict* sd = nc_module_state_dict(m);
    if(!sd) return NC_ERR_ALLOC;

    nc_error err = nc_state_dict_save(sd, path);
    nc_state_dict_free(sd);

    return err;
}

nc_error nc_module_load(nc_module* m, const char* path) {
    if(!m || !path) return NC_ERR_NULL_PTR;

    nc_state_dict* sd = nc_state_dict_load(path);
    if(!sd) return NC_ERR_ALLOC;

    nc_error err = nc_module_load_state_dict(m, sd);
    nc_state_dict_free(sd);

    return err;
}

nc_error nc_module_save_desc(nc_module* m, const char* path, const char* description) {
    if(!m || !path) return NC_ERR_NULL_PTR;

        nc_state_dict* sd = nc_module_state_dict(m);
    if (!sd) return NC_ERR_ALLOC;
    
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        nc_state_dict_free(sd);
        return NC_ERR_IO;
    }
    
    nc_file_header header = {
        .magic = NC_FILE_MAGIC,
        .version = NC_FILE_VERSION,
        .n_tensors = (uint32_t)sd->n_tensors,
        .flags = 0
    };
    if (description) {
        strncpy(header.description, description, sizeof(header.description) - 1);
    }
    
    if (!write_header(fp, &header)) {
        fclose(fp);
        nc_state_dict_free(sd);
        return NC_ERR_IO;
    }
    
    for (size_t i = 0; i < sd->n_tensors; i++) {
        nc_error err = nc_tensor_save_fp(sd->tensors[i], fp, sd->names[i]);
        if (err != NC_OK) {
            fclose(fp);
            nc_state_dict_free(sd);
            return err;
        }
    }
    
    fclose(fp);
    nc_state_dict_free(sd);
    return NC_OK;
}

// ============================================
// Checkpoint
// ============================================

nc_error nc_checkpoint_save(nc_checkpoint* ckpt, const char* path) {
    if (!ckpt || !path) return NC_ERR_NULL_PTR;
    
    FILE* fp = fopen(path, "wb");
    if (!fp) return NC_ERR_IO;
    
    // Count total tensors
    uint32_t n_model = ckpt->model_state ? (uint32_t)ckpt->model_state->n_tensors : 0;
    uint32_t n_optim = ckpt->optimizer_state ? (uint32_t)ckpt->optimizer_state->n_tensors : 0;
    
    // Write header
    nc_file_header header = {
        .magic = NC_FILE_MAGIC,
        .version = NC_FILE_VERSION,
        .n_tensors = n_model + n_optim,
        .flags = 1  // Flag indicating checkpoint format
    };
    snprintf(header.description, sizeof(header.description), 
             "Checkpoint epoch=%zu loss=%.6f", ckpt->epoch, ckpt->loss);
    
    if (!write_header(fp, &header)) {
        fclose(fp);
        return NC_ERR_IO;
    }
    
    // Write checkpoint metadata
    fwrite(&ckpt->epoch, sizeof(size_t), 1, fp);
    fwrite(&ckpt->loss, sizeof(double), 1, fp);
    fwrite(&n_model, sizeof(uint32_t), 1, fp);
    fwrite(&n_optim, sizeof(uint32_t), 1, fp);
    
    // Write model state
    if (ckpt->model_state) {
        for (size_t i = 0; i < ckpt->model_state->n_tensors; i++) {
            nc_error err = nc_tensor_save_fp(
                ckpt->model_state->tensors[i], fp, 
                ckpt->model_state->names[i]);
            if (err != NC_OK) {
                fclose(fp);
                return err;
            }
        }
    }
    
    // Write optimizer state
    if (ckpt->optimizer_state) {
        for (size_t i = 0; i < ckpt->optimizer_state->n_tensors; i++) {
            nc_error err = nc_tensor_save_fp(
                ckpt->optimizer_state->tensors[i], fp,
                ckpt->optimizer_state->names[i]);
            if (err != NC_OK) {
                fclose(fp);
                return err;
            }
        }
    }
    
    fclose(fp);
    return NC_OK;
}

nc_checkpoint* nc_checkpoint_load(const char* path) {
    if (!path) return NULL;
    
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    
    nc_file_header header;
    if (!read_header(fp, &header) || header.flags != 1) {
        fclose(fp);
        return NULL;
    }
    
    nc_checkpoint* ckpt = nc_calloc(1, sizeof(nc_checkpoint));
    if (!ckpt) {
        fclose(fp);
        return NULL;
    }
    
    // Read metadata
    uint32_t n_model, n_optim;
    fread(&ckpt->epoch, sizeof(size_t), 1, fp);
    fread(&ckpt->loss, sizeof(double), 1, fp);
    fread(&n_model, sizeof(uint32_t), 1, fp);
    fread(&n_optim, sizeof(uint32_t), 1, fp);
    
    // Read model state
    if (n_model > 0) {
        ckpt->model_state = nc_state_dict_create();
        for (uint32_t i = 0; i < n_model; i++) {
            char name[NC_MAX_NAME_LEN] = {0};
            nc_tensor* t = nc_tensor_load_fp(fp, name);
            if (t) nc_state_dict_add(ckpt->model_state, name, t);
        }
    }
    
    // Read optimizer state
    if (n_optim > 0) {
        ckpt->optimizer_state = nc_state_dict_create();
        for (uint32_t i = 0; i < n_optim; i++) {
            char name[NC_MAX_NAME_LEN] = {0};
            nc_tensor* t = nc_tensor_load_fp(fp, name);
            if (t) nc_state_dict_add(ckpt->optimizer_state, name, t);
        }
    }
    
    fclose(fp);
    return ckpt;
}

void nc_checkpoint_free(nc_checkpoint* ckpt) {
    if (!ckpt) return;
    if (ckpt->model_state) nc_state_dict_free(ckpt->model_state);
    if (ckpt->optimizer_state) nc_state_dict_free(ckpt->optimizer_state);
    nc_free(ckpt);
}

// ============================================
// Utility
// ============================================

void nc_file_info(const char* path) {
    if (!path) return;
    
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        printf("Cannot open file: %s\n", path);
        return;
    }
    
    nc_file_header header;
    if (!read_header(fp, &header)) {
        printf("Invalid Nocta file format\n");
        fclose(fp);
        return;
    }
    
    printf("=== Nocta File Info ===\n");
    printf("File: %s\n", path);
    printf("Version: %u\n", header.version);
    printf("Tensors: %u\n", header.n_tensors);
    printf("Description: %s\n", header.description);
    printf("\nTensors:\n");
    
    for (uint32_t i = 0; i < header.n_tensors; i++) {
        nc_tensor_header th;
        if (!read_tensor_header(fp, &th)) break;
        
        printf("  [%u] %s: ", i, th.name);
        printf("dtype=%s, shape=(", nc_dtype_to_string((nc_dtype)th.dtype));
        for (uint32_t d = 0; d < th.ndim; d++) {
            printf("%lu%s", (unsigned long)th.shape[d], d < th.ndim - 1 ? ", " : "");
        }
        printf("), size=%lu bytes\n", (unsigned long)th.data_size);
        
        // Skip tensor data
        fseek(fp, (long)th.data_size, SEEK_CUR);
    }
    
    fclose(fp);
}

bool nc_file_verify(const char* path) {
    if (!path) return false;
    
    FILE* fp = fopen(path, "rb");
    if (!fp) return false;
    
    nc_file_header header;
    if (!read_header(fp, &header)) {
        fclose(fp);
        return false;
    }
    
    // Verify each tensor can be read
    for (uint32_t i = 0; i < header.n_tensors; i++) {
        nc_tensor_header th;
        if (!read_tensor_header(fp, &th)) {
            fclose(fp);
            return false;
        }
        
        // Verify we can seek past the data
        if (fseek(fp, (long)th.data_size, SEEK_CUR) != 0) {
            fclose(fp);
            return false;
        }
    }
    
    fclose(fp);
    return true;
}