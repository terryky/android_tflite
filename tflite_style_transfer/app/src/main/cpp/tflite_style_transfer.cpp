/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_style_transfer.h"
#include "util_debug.h"


static tflite_interpreter_t s_interpreter_style_predict;
static float    *s_style_predict_in_ptr;
static int      s_style_predict_img_w = 0;
static int      s_style_predict_img_h = 0;


static tflite_interpreter_t s_interpreter_style_transfer;
static float    *s_style_transfer_style_in_ptr;
static float    *s_style_transfer_content_in_ptr;
static int      s_style_transfer_style_dim;
static int      s_style_transfer_img_w = 0;
static int      s_style_transfer_img_h = 0;




static int
init_tflite_style_predict(tflite_interpreter_t *p, const char *model_buf, size_t model_size)
{
    tflite_create_interpreter (p, model_buf, model_size);

    s_style_predict_in_ptr  = p->interpreter->typed_input_tensor<float>(0);

    /* input image dimention */
    int input_idx = p->interpreter->inputs()[0];
    TfLiteIntArray *in_dim = p->interpreter->tensor(input_idx)->dims;
    s_style_predict_img_w = in_dim->data[2];
    s_style_predict_img_h = in_dim->data[1];
    DBG_LOGI ("input image size: (%d, %d)\n", s_style_predict_img_w, s_style_predict_img_h);

    /* style output  dimention */
    int output_idx = p->interpreter->outputs()[0];
    TfLiteIntArray *out_dim = p->interpreter->tensor(output_idx)->dims;
    int s0 = out_dim->data[0];
    int s1 = out_dim->data[1];
    int s2 = out_dim->data[2];
    int s3 = out_dim->data[3];
    DBG_LOGI ("style output dim: (%dx%dx%dx%d)\n", s0, s1, s2, s3);

    return 0;
}

static int
init_tflite_style_trans(tflite_interpreter_t *p, const char *model_buf, size_t model_size)
{
    tflite_create_interpreter (p, model_buf, model_size);

    s_style_transfer_content_in_ptr = p->interpreter->typed_input_tensor<float>(0);
    s_style_transfer_style_in_ptr   = p->interpreter->typed_input_tensor<float>(1);

    /* input image dimention */
    int input_content_idx = p->interpreter->inputs()[0];
    TfLiteIntArray *in_content_dim = p->interpreter->tensor(input_content_idx)->dims;
    s_style_transfer_img_w = in_content_dim->data[2];
    s_style_transfer_img_h = in_content_dim->data[1];
    DBG_LOGI ("input image size: [%d](%d, %d)\n", input_content_idx, s_style_transfer_img_w, s_style_transfer_img_h);

    int input_style_idx = p->interpreter->inputs()[1];
    TfLiteIntArray *in_style_dim = p->interpreter->tensor(input_style_idx)->dims;
    s_style_transfer_style_dim = in_style_dim->data[3];
    DBG_LOGI ("input image size: [%d](%d)\n", input_style_idx, s_style_transfer_style_dim);

    int output_style_idx = p->interpreter->outputs()[0];
    TfLiteIntArray *out_dim = p->interpreter->tensor(output_style_idx)->dims;
    int s0 = out_dim->data[0];
    int s1 = out_dim->data[1];
    int s2 = out_dim->data[2];
    int s3 = out_dim->data[3];
    DBG_LOGI ("style output dim: (%dx%dx%dx%d)\n", s0, s1, s2, s3);

    return 0;
}


int
init_tflite_style_transfer (const char *predict_model_buf, size_t predict_model_size,
                            const char *transfr_model_buf, size_t transfr_model_size)
{
    init_tflite_style_predict (&s_interpreter_style_predict,  predict_model_buf, predict_model_size);
    init_tflite_style_trans   (&s_interpreter_style_transfer, transfr_model_buf, transfr_model_size);

    return 0;
}

void *
get_style_predict_input_buf (int *w, int *h)
{
    *w = s_style_predict_img_w;
    *h = s_style_predict_img_h;
    return s_style_predict_in_ptr;
}

void *
get_style_transfer_style_input_buf (int *size)
{
    s_style_transfer_style_in_ptr = s_interpreter_style_transfer.interpreter->typed_input_tensor<float>(1);

    *size = s_style_transfer_style_dim;
    return s_style_transfer_style_in_ptr;
}

void *
get_style_transfer_content_input_buf (int *w, int *h)
{
    s_style_transfer_content_in_ptr = s_interpreter_style_transfer.interpreter->typed_input_tensor<float>(0);

    *w = s_style_transfer_img_w;
    *h = s_style_transfer_img_h;
    return s_style_transfer_content_in_ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_style_predict (style_predict_t *predict_result)
{
    if (s_interpreter_style_predict.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int output_idx = s_interpreter_style_predict.interpreter->outputs()[0];
    TfLiteIntArray *out_dim = s_interpreter_style_predict.interpreter->tensor(output_idx)->dims;
    //int s0 = out_dim->data[0];
    //int s1 = out_dim->data[1];
    //int s2 = out_dim->data[2];
    int s3 = out_dim->data[3];

    predict_result->size  = s3;
    predict_result->param = s_interpreter_style_predict.interpreter->typed_output_tensor<float>(0); 
    
    return 0;
}


int
invoke_style_transfer (style_transfer_t *transfered_result)
{
    if (s_interpreter_style_transfer.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int output_idx = s_interpreter_style_transfer.interpreter->outputs()[0];
    TfLiteIntArray *out_dim = s_interpreter_style_transfer.interpreter->tensor(output_idx)->dims;
    //int s0 = out_dim->data[0];
    int s1 = out_dim->data[1];
    int s2 = out_dim->data[2];
    //int s3 = out_dim->data[3];
    
    transfered_result->h = s1;
    transfered_result->w = s2;
    transfered_result->img = s_interpreter_style_transfer.interpreter->typed_output_tensor<float>(0); 

    return 0;
}
