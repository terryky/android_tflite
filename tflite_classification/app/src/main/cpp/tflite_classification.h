/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_CLASSIFICATION_H_
#define TFLITE_CLASSIFICATION_H_

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * https://www.tensorflow.org/lite/guide/hosted_models
 */
#define CLASSIFY_MODEL_PATH        "classification_model/mobilenet_v1_1.0_224.tflite"
#define CLASSIFY_QUANT_MODEL_PATH  "classification_model/mobilenet_v1_1.0_224_quant.tflite"
#define CLASSIFY_LABEL_MAP_PATH    "classification_model/class_label.txt"

#define MAX_CLASS_NUM  1001

typedef struct _classify_t
{
    int     id;
    float   score;
    char    name[64];
} classify_t;

typedef struct _classification_result_t
{
    int num;
    classify_t classify[MAX_CLASS_NUM];
} classification_result_t;



int   init_tflite_classification (const char *model_buf, size_t model_size, 
                                  const char *label_buf, size_t label_size);
int   get_classification_input_type ();
void  *get_classification_input_buf (int *w, int *h);

int   invoke_classification (classification_result_t *class_result);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_CLASSIFICATION_H_ */
