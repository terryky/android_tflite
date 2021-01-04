/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_ANIMEGAN2_H_
#define TFLITE_ANIMEGAN2_H_

#ifdef __cplusplus
extern "C" {
#endif

#define ANIMEGAN2_MODEL_PATH        "model/animeganv2_hayao_256x256.tflite"
#define ANIMEGAN2_QUANT_MODEL_PATH  "model/animeganv2_hayao_256x256_integer_quant.tflite"

typedef struct _animegan2_t
{
    int w, h;
    void *param;
} animegan2_t;


int init_tflite_animegan2 (const char *model_buf, size_t model_size);
void  *get_animegan2_input_buf (int *w, int *h);

int  invoke_animegan2 (animegan2_t *animegan2_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_ANIMEGAN2_H_ */
