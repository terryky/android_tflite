/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_STYLE_TRANSFER_H_
#define TFLITE_STYLE_TRANSFER_H_

#ifdef __cplusplus
extern "C" {
#endif

#if 1 /* for GPU Delegate */
#define STYLE_PREDICT_MODEL_PATH  "style_transfer_model/style_predict_f16_256.tflite"
#define STYLE_TRANSFER_MODEL_PATH "style_transfer_model/style_transfer_f16_384.tflite"
#else
#define STYLE_PREDICT_MODEL_PATH  "style_transfer_model/style_predict_quantized_256.tflite"
#define STYLE_TRANSFER_MODEL_PATH "style_transfer_model/style_transfer_quantized_384.tflite"
#endif

typedef struct _style_predict_t
{
    int size;
    void *param;
} style_predict_t;

typedef struct _style_transfer_t
{
    int w, h;
    void *img;
} style_transfer_t;


int   init_tflite_style_transfer (const char *predict_model_buf, size_t predict_model_size,
                                  const char *transfr_model_buf, size_t transfr_model_size);
void  *get_style_predict_input_buf (int *w, int *h);
void  *get_style_transfer_style_input_buf (int *size);
void  *get_style_transfer_content_input_buf (int *w, int *h);

int invoke_style_predict (style_predict_t  *predict_result);
int invoke_style_transfer(style_transfer_t *transfer_result);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_STYLE_TRANSFER_H_ */
