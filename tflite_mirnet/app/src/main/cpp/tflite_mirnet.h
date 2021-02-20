/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_MIRNET_H_
#define TFLITE_MIRNET_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MIRNET_MODEL_PATH        "model/lite-model_mirnet-fixed_fp16_1.tflite"
#define MIRNET_QUANT_MODEL_PATH  "model/lite-model_mirnet-fixed_integer_1.tflite"

typedef struct _mirnet_t
{
    int w, h;
    void *param;
} mirnet_t;


int   init_tflite_mirnet (const char *model_buf, size_t model_size);
void  *get_mirnet_input_buf (int *w, int *h);

int  invoke_mirnet (mirnet_t *mirnet_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_MIRNET_H_ */
