/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_BLAZEFACE_H_
#define TFLITE_BLAZEFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * https://github.com/PINTO0309/PINTO_model_zoo/tree/master/041_DBFace/01_float32
 * https://github.com/PINTO0309/PINTO_model_zoo/tree/master/041_DBFace/03_integer_quantization
 */
//#define DBFACE_MODEL_PATH        "./model/dbface_keras_256x256_float32_nhwc.tflite"
//#define DBFACE_QUANT_MODEL_PATH  "./model/dbface_keras_256x256_integer_quant_nhwc.tflite"

#define DBFACE_MODEL_PATH        "model/dbface_keras_480x640_float32_nhwc.tflite"
#define DBFACE_QUANT_MODEL_PATH  "model/dbface_keras_480x640_integer_quant_nhwc.tflite"

#define MAX_FACE_NUM  100

enum face_key_id {
    kRightEye = 0,  //  0
    kLeftEye,       //  1
    kNose,          //  2
    kMouth,         //  3
    kRightEar,      //  4

    kFaceKeyNum
};

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct _face_t
{
    float score;
    fvec2 topleft;
    fvec2 btmright;
    fvec2 keys[kFaceKeyNum];
} face_t;

typedef struct _dbface_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} dbface_result_t;

typedef struct _dbface_config_t
{
    float score_thresh;
    float iou_thresh;
} dbface_config_t;

int init_tflite_dbface (const char *model_buf, size_t model_size, dbface_config_t *config);

void *get_dbface_input_buf (int *w, int *h);
int invoke_dbface (dbface_result_t *dbface_result, dbface_config_t *config);

#ifdef __cplusplus
}
#endif

#endif /* TRT_DBFACE_H_ */
