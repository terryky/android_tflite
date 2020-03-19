/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_TFLITE_H_
#define _UTIL_TFLITE_H_

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#if defined (USE_GPU_DELEGATEV2)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#include "util_debug.h"



using namespace std;
using namespace tflite;

typedef struct tflite_interpreter_t
{
    unique_ptr<FlatBufferModel>     model;
    unique_ptr<Interpreter>         interpreter;
    ops::builtin::BuiltinOpResolver resolver;
} tflite_interpreter_t;


int tflite_create_interpreter (tflite_interpreter_t *p, const char *model_buf, size_t model_size);

#endif /* _UTIL_TFLITE_H_ */
