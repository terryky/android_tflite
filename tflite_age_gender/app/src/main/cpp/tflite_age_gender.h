/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_AGE_GENDER_H_
#define TFLITE_AGE_GENDER_H_

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite
 * https://github.com/yu4u/age-gender-estimation"
 */
#define FACE_DETECT_MODEL_PATH       "model/face_detection_front.tflite"
#define AGE_GENDER_MODEL_PATH        "model/EfficientNetB3_224_weights.11-3.44.tflite"

#define MAX_FACE_NUM  100

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

typedef struct _age_t
{
    int   age;
    float score;
} age_t;

typedef struct _gender_t
{
    float score_m;
    float score_f;
} gender_t;

typedef struct _face_detect_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} face_detect_result_t;


typedef struct _age_gender_result_t
{
    age_t    age;
    gender_t gender;
} age_gender_result_t;



int init_tflite_age_gender (const char *face_detect_model_buf, size_t face_detect_model_size, 
                            const char *age_gender_model_buf, size_t age_gender_model_size);

void *get_face_detect_input_buf (int *w, int *h);
int  invoke_face_detect (face_detect_result_t *facedet_result);

void  *get_age_gender_input_buf (int *w, int *h);
int invoke_age_gender (age_gender_result_t *age_gender_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_AGE_GENDER_H_ */
