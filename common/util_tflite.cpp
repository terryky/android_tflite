/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"



void
tflite_print_tensor_dim (unique_ptr<Interpreter> &interpreter, int tensor_id)
{
    TfLiteIntArray *dim = interpreter->tensor(tensor_id)->dims;

    for (int i = 0; i < dim->size; i ++)
    {
        if (i > 0)
            DBG_LOGI ("x");
        DBG_LOGI ("%d", dim->data[i]);
    }
    DBG_LOGI ("\n");
}

void
tflite_print_tensor_info (unique_ptr<Interpreter> &interpreter)
{
    int i, idx;
    int in_size  = interpreter->inputs().size();
    int out_size = interpreter->outputs().size();

    DBG_LOGI ("-----------------------------------------------------------------------------\n");
    DBG_LOGI ("tensors size     : %zu\n", interpreter->tensors_size());
    DBG_LOGI ("nodes   size     : %zu\n", interpreter->nodes_size());
    DBG_LOGI ("number of inputs : %d\n", in_size);
    DBG_LOGI ("number of outputs: %d\n", out_size);
    DBG_LOGI ("input(0) name    : %s\n", interpreter->GetInputName(0));

    DBG_LOGI ("\n");
    DBG_LOGI ("-----------------------------------------------------------------------------\n");
    DBG_LOGI ("                     name                     bytes  type  scale   zero_point\n");
    DBG_LOGI ("-----------------------------------------------------------------------------\n");
    int t_size = interpreter->tensors_size();
    for (i = 0; i < t_size; i++) 
    {
        DBG_LOGI ("Tensor[%2d] %-32s %8zu, %2d, %f, %3d\n", i,
            interpreter->tensor(i)->name, 
            interpreter->tensor(i)->bytes,
            interpreter->tensor(i)->type,
            interpreter->tensor(i)->params.scale,
            interpreter->tensor(i)->params.zero_point);
    }

    DBG_LOGI ("\n");
    DBG_LOGI ("-----------------------------------------------------------------------------\n");
    DBG_LOGI (" Input Tensor Dimension\n");
    DBG_LOGI ("-----------------------------------------------------------------------------\n");
    for (i = 0; i < in_size; i ++)
    {
        idx = interpreter->inputs()[i];
        DBG_LOGI ("Tensor[%2d]: ", idx);
        tflite_print_tensor_dim (interpreter, idx);
    }

    DBG_LOGI ("\n");
    DBG_LOGI ("-----------------------------------------------------------------------------\n");
    DBG_LOGI (" Output Tensor Dimension\n");
    DBG_LOGI ("-----------------------------------------------------------------------------\n");
    for (i = 0; i < out_size; i ++)
    {
        idx = interpreter->outputs()[i];
        DBG_LOGI ("Tensor[%2d]: ", idx);
        tflite_print_tensor_dim (interpreter, idx);
    }

    DBG_LOGI ("\n");
    DBG_LOGI ("-----------------------------------------------------------------------------\n");
    PrintInterpreterState(interpreter.get());
    DBG_LOGI ("-----------------------------------------------------------------------------\n");
}



int
tflite_create_interpreter (tflite_interpreter_t *p, const char *model_buf, size_t model_size)
{
    p->model = FlatBufferModel::BuildFromBuffer(model_buf, model_size);
    if (!p->model)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*(p->model), p->resolver)(&(p->interpreter));
    if (!p->interpreter)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if defined (USE_GL_DELEGATE)
    const TfLiteGpuDelegateOptions options = {
        .metadata = NULL,
        .compile_options = {
            .precision_loss_allowed = 1,  // FP16
            .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
            .dynamic_batch_enabled = 0,   // Not fully functional yet
        },
    };
    auto* delegate = TfLiteGpuDelegateCreate(&options);

    if (p->interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
#endif

#if defined (USE_GPU_DELEGATEV2)
    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, // FP16
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    };
    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    if (p->interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
#endif

    p->interpreter->SetNumThreads(4);
    if (p->interpreter->AllocateTensors() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    tflite_print_tensor_info (p->interpreter);
#endif

    return 0;
}


