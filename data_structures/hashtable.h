#ifndef SIMPLE_HASHTABLE_H
#define SIMPLE_HASHTABLE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ======= CONFIG =======
// Load factor before resize (0.70 = 70%)
#define HT_LOAD_FACTOR 0.70

// ======= DATA STRUCTURES =======
typedef struct {
    char *key;
    void *value;
    uint64_t count;
    int in_use;
} HT_Entry;

typedef struct {
    HT_Entry *entries;
    size_t capacity;
    size_t count;
} HashTable;

// ======= FNV-1a 64-bit HASH =======
static inline uint64_t ht_hash_str(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

// ======= INTERNAL: RESIZE =======
static int ht_resize(HashTable *ht, size_t new_cap);

// ======= CREATE =======
static inline HashTable *ht_create(size_t initial_capacity) {
    HashTable *ht = (HashTable *)calloc(1, sizeof(HashTable));
    ht->capacity = (initial_capacity < 8) ? 8 : initial_capacity;
    ht->entries = (HT_Entry *)calloc(ht->capacity, sizeof(HT_Entry));
    return ht;
}

// ======= FREE =======
static inline void ht_free(HashTable *ht, int free_values) {
    for (size_t i = 0; i < ht->capacity; i++) {
        if (ht->entries[i].in_use) {
            free(ht->entries[i].key);
            if (free_values && ht->entries[i].value) {
                free(ht->entries[i].value);
            }
        }
    }
    free(ht->entries);
    free(ht);
}

// ======= INSERT / UPDATE =======
static inline int ht_set(HashTable *ht, const char *key, void *value) {
    if ((double)ht->count / ht->capacity >= HT_LOAD_FACTOR) {
        ht_resize(ht, ht->capacity * 2);
    }

    uint64_t hash = ht_hash_str(key);
    size_t idx = hash % ht->capacity;

    while (ht->entries[idx].in_use) {
        if (strcmp(ht->entries[idx].key, key) == 0) {
            ht->entries[idx].value = value;
            ++ht->entries[idx].count;
            return 1;
        }
        idx = (idx + 1) % ht->capacity;
    }

    ht->entries[idx].key = strdup(key);
    ht->entries[idx].value = value;
    ht->entries[idx].in_use = 1;
    ht->entries[idx].count = 1;
    ht->count++;
    return 1;
}

// ======= LOOKUP =======
static inline void *ht_get(HashTable *ht, const char *key) {
    uint64_t hash = ht_hash_str(key);
    size_t idx = hash % ht->capacity;

    while (ht->entries[idx].in_use) {
        if (strcmp(ht->entries[idx].key, key) == 0) {
            return ht->entries[idx].value;
        }
        idx = (idx + 1) % ht->capacity;
    }
    return NULL;
}

// ======= REMOVE =======
static inline int ht_remove(HashTable *ht, const char *key) {
    uint64_t hash = ht_hash_str(key);
    size_t idx = hash % ht->capacity;

    while (ht->entries[idx].in_use) {
        if (strcmp(ht->entries[idx].key, key) == 0) {
            free(ht->entries[idx].key);
            ht->entries[idx].key = NULL;
            ht->entries[idx].value = NULL;
            ht->entries[idx].in_use = 0;
            ht->count--;
            return 1;
        }
        idx = (idx + 1) % ht->capacity;
    }
    return 0;
}

// ======= INTERNAL: RESIZE IMPL =======
static int ht_resize(HashTable *ht, size_t new_cap) {
    HT_Entry *old_entries = ht->entries;
    size_t old_cap = ht->capacity;

    ht->entries = (HT_Entry *)calloc(new_cap, sizeof(HT_Entry));
    ht->capacity = new_cap;
    ht->count = 0;

    for (size_t i = 0; i < old_cap; i++) {
        if (old_entries[i].in_use) {
            ht_set(ht, old_entries[i].key, old_entries[i].value);
            free(old_entries[i].key);
        }
    }
    free(old_entries);
    return 1;
}

#endif // SIMPLE_HASHTABLE_H

