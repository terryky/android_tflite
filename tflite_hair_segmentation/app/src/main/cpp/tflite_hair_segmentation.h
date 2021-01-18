/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_HAIR_SEGMENTATION_H_
#define TFLITE_HAIR_SEGMENTATION_H_

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/models/hair_segmentation.tflite
 */
#define SEGMENTATION_MODEL_PATH  "hair_segmentation_model/hair_segmentation.tflite"

#define MAX_SEGMENT_CLASS 2


typedef struct _segmentation_result_t
{
    float *segmentmap;
    int   segmentmap_dims[3];
} segmentation_result_t;



int   init_tflite_segmentation (const char *model_buf, size_t model_size);
void  *get_segmentation_input_buf (int *w, int *h);

int invoke_segmentation (segmentation_result_t *segment_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_HAIR_SEGMENTATION_H_ */
