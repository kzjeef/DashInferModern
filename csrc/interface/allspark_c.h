/*!
 * Copyright (c) 2026 Segno System.
 * @file    allspark_c.h
 */

#ifndef ALLSPARK_C_H_
#define ALLSPARK_C_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(ALLSPARK_C_EXPORTS)
#define AS_C_API __declspec(dllexport)
#else
#define AS_C_API __declspec(dllimport)
#endif
#else
#define AS_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define AS_C_API_VERSION 1u

typedef int32_t as_status_t;
typedef struct as_engine as_engine_t;

enum as_status_code {
  AS_STATUS_SUCCESS = 0,
  AS_STATUS_UNKNOWN_ERROR = 1,
  AS_STATUS_PARAM_ERROR = 2,
  AS_STATUS_IO_ERROR = 3,
  AS_STATUS_MEMORY_ERROR = 4,
  AS_STATUS_RUNTIME_ERROR = 5,
  AS_STATUS_EXCEED_LIMIT = 7,
  AS_STATUS_INVALID_CALL = 8,
  AS_STATUS_EMPTY_REQUEST = 9,
  AS_STATUS_ILLEGAL_REQUEST_ID = 10,
  AS_STATUS_CACHE_MEMORY_OUT = 11,
  AS_STATUS_REQUEST_DENIED = 12,
  AS_STATUS_CHUNK_PREFILL = 14,
  AS_STATUS_DEPRECATED = 20,
  AS_STATUS_LORA_NUM_EXCEED_LIMIT = 21,
  AS_STATUS_LORA_RANK_EXCEED_LIMIT = 22,
  AS_STATUS_LORA_NOT_FOUND = 23,
  AS_STATUS_LORA_ALREADY_LOADED = 24,
  AS_STATUS_LORA_IN_USE = 25,
  AS_STATUS_STREAMING = 200,
};

/** Allocate an engine and return it through `engine`. */
AS_C_API as_status_t as_engine_create(as_engine_t** engine);

/** Destroy an engine. Passing NULL is allowed. */
AS_C_API void as_engine_destroy(as_engine_t* engine);

/**
 * Copy the engine version into `buffer`.
 *
 * Set `buffer` to NULL to query the required byte count, including the final
 * NUL. On entry `*buffer_size` is the available byte count; on return it is
 * always the required byte count.
 */
AS_C_API as_status_t as_engine_get_version(as_engine_t* engine, char* buffer,
                                           size_t* buffer_size);

/** Return a static name for a status code. */
AS_C_API const char* as_status_string(as_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ALLSPARK_C_H_
