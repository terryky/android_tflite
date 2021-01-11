/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_SELFIE2ANIME_H_
#define TFLITE_SELFIE2ANIME_H_

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite
 * https://github.com/margaretmz/selfie2anime-e2e-tutorial/blob/master/android/selfie2anime/app/src/main/ml/selfie2anime.tflite
 */
#define FACE_DETECT_MODEL_PATH      "model/face_detection_front.tflite"
#define SELFIE2ANIME_MODEL_PATH     "model/selfie2anime.tflite"

#define MAX_FACE_NUM     100

enum face_key_id {
    kRightEye = 0,  //  0
    kLeftEye,       //  1
    kNose,          //  2
    kMouth,         //  3
    kRightEar,      //  4
    kLeftEar,       //  5

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

    float rotation;
    float face_cx;
    float face_cy;
    float face_w;
    float face_h;
    fvec2 face_pos[4];
} face_t;

typedef struct _face_detect_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} face_detect_result_t;


typedef struct _selfie2anime_result_t
{
    float *segmentmap;
    int   segmentmap_dims[3]; 
} selfie2anime_result_t;



int init_tflite_selfie2anime (const char *face_detect_model_buf, size_t face_detect_model_size, 
                              const char *face_anime_model_buf,  size_t face_anime_model_size);

void *get_face_detect_input_buf (int *w, int *h);
int  invoke_face_detect (face_detect_result_t *facedet_result);

void  *get_selfie2anime_input_buf (int *w, int *h);
int invoke_selfie2anime (selfie2anime_result_t *selfie2anime_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_SELFIE2ANIME_H_ */
