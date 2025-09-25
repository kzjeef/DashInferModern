/*!
 * Copyright (c) 2026 Segno System.
 * @file    mini_inference.c
 */

#include <allspark_c.h>

#include <stdio.h>
#include <stdlib.h>

static int check(as_status_t status, const char* operation) {
  if (status == AS_STATUS_SUCCESS) {
    return 1;
  }
  fprintf(stderr, "%s failed: %s (%d)\n", operation,
          as_status_string(status), status);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL.asgraph MODEL.asparam [CUDA:0]\n",
            argv[0]);
    return 2;
  }

  int exit_code = 1;
  int model_started = 0;
  as_engine_t* engine = NULL;
  as_request_t* request = NULL;
  const char* model_name = "deepseek_v3_mini";

  if (!check(as_engine_create(&engine), "create engine")) {
    goto cleanup;
  }

  as_model_config_t model_config;
  as_model_config_init(&model_config);
  model_config.model_name = model_name;
  model_config.model_path = argv[1];
  model_config.weights_path = argv[2];
  model_config.compute_unit = argc > 3 ? argv[3] : "CUDA:0";
  model_config.engine_max_length = 64;
  model_config.engine_max_batch = 1;

  if (!check(as_engine_build_model(engine, &model_config), "build model")) {
    goto cleanup;
  }
  if (!check(as_engine_start_model(engine, model_name), "start model")) {
    goto cleanup;
  }
  model_started = 1;

  int64_t input_values[] = {1, 42, 7, 2};
  int64_t input_shape[] = {1, 4};
  DLManagedTensor input_ids = {0};
  input_ids.dl_tensor.data = input_values;
  input_ids.dl_tensor.device.device_type = kDLCPU;
  input_ids.dl_tensor.device.device_id = 0;
  input_ids.dl_tensor.ndim = 2;
  input_ids.dl_tensor.dtype.code = kDLInt;
  input_ids.dl_tensor.dtype.bits = 64;
  input_ids.dl_tensor.dtype.lanes = 1;
  input_ids.dl_tensor.shape = input_shape;

  as_named_tensor_t inputs[] = {{"input_ids", &input_ids}};
  as_generate_config_t generate_config;
  as_generate_config_init(&generate_config);
  generate_config.max_length = 16;
  generate_config.top_k = 1;
  generate_config.top_p = 0.0f;

  if (!check(as_engine_start_request(engine, model_name, inputs, 1,
                                     &generate_config, &request),
             "start request")) {
    goto cleanup;
  }

  for (;;) {
    size_t token_count = 0;
    as_status_t status =
        as_request_fetch_tokens(request, 1000, NULL, &token_count);
    if (status == AS_STATUS_SUCCESS && token_count > 0) {
      int64_t* tokens = (int64_t*)malloc(token_count * sizeof(*tokens));
      if (tokens == NULL) {
        fprintf(stderr, "token allocation failed\n");
        goto cleanup;
      }
      size_t capacity = token_count;
      status = as_request_fetch_tokens(request, 0, tokens, &capacity);
      if (!check(status, "fetch tokens")) {
        free(tokens);
        goto cleanup;
      }
      for (size_t i = 0; i < capacity; ++i) {
        printf("%lld%c", (long long)tokens[i], i + 1 == capacity ? '\n' : ' ');
      }
      free(tokens);
    } else if (status != AS_STATUS_EMPTY_REQUEST &&
               status != AS_STATUS_SUCCESS) {
      check(status, "poll tokens");
      goto cleanup;
    }

    int32_t request_status = AS_REQUEST_INIT;
    if (!check(as_request_get_status(request, &request_status),
               "request status")) {
      goto cleanup;
    }
    if (request_status == AS_REQUEST_FINISHED ||
        request_status == AS_REQUEST_INTERRUPTED ||
        request_status == AS_REQUEST_INTERNAL_ERROR) {
      break;
    }
  }

  if (!check(as_engine_sync_request(engine, request), "sync request")) {
    goto cleanup;
  }
  exit_code = 0;

cleanup:
  if (request != NULL) {
    if (exit_code != 0) {
      as_engine_stop_request(engine, request);
    }
    as_engine_release_request(engine, request);
  }
  if (model_started) {
    as_engine_stop_model(engine, model_name);
    as_engine_release_model(engine, model_name);
  }
  as_engine_destroy(engine);
  return exit_code;
}
