/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#if defined (USE_GPU_DELEGATEV2)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#include <list>
#include "tflite_blazeface.h"
#include "util_debug.h"
#include <float.h>

using namespace std;
using namespace tflite;

unique_ptr<FlatBufferModel> model;
unique_ptr<Interpreter> interpreter;
ops::builtin::BuiltinOpResolver resolver;

static float   *in_ptr;
static float   *scores_ptr;
static float   *bboxes_ptr;

static int     s_img_w = 0;
static int     s_img_h = 0;
static std::list<fvec2> s_anchors;


/*
 * determine where the anchor points are scatterd.
 *   https://github.com/tensorflow/tfjs-models/blob/master/blazeface/src/face.ts
 */
static int
create_blazeface_anchors(int input_w, int input_h)
{
    /* ANCHORS_CONFIG  */
    int strides[2] = {8, 16};
    int anchors[2] = {2,  6};

    int numtotal = 0;

    for (int i = 0; i < 2; i ++)
    {
        int stride = strides[i];
        int gridCols = (input_w + stride -1) / stride;
        int gridRows = (input_h + stride -1) / stride;
        int anchorNum = anchors[i];

        fvec2 anchor;
        for (int gridY = 0; gridY < gridRows; gridY ++)
        {
            anchor.y = stride * (gridY + 0.5f);
            for (int gridX = 0; gridX < gridCols; gridX ++)
            {
                anchor.x = stride * (gridX + 0.5f);
                for (int n = 0; n < anchorNum; n ++)
                {
                    s_anchors.push_back (anchor);
                    numtotal ++;
                }
            }
        }
    }
    return numtotal;
}


int
init_tflite_blazeface(const char *model_buf, size_t model_size)
{
    model = FlatBufferModel::BuildFromBuffer (model_buf, model_size);
    if (!model)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -2;
    }

#if defined (USE_GPU_DELEGATEV2)
    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, // FP16
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    };
    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
#endif

    interpreter->SetNumThreads(4);
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -3;
    }

    in_ptr     = interpreter->typed_input_tensor<float>(0);
    bboxes_ptr = interpreter->typed_output_tensor<float>(0);
    scores_ptr = interpreter->typed_output_tensor<float>(1);

    /* input image dimention */
    int input_idx = interpreter->inputs()[0];
    TfLiteIntArray *dim = interpreter->tensor(input_idx)->dims;
    s_img_w = dim->data[2];
    s_img_h = dim->data[1];
    DBG_LOGI ("input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* scores dimention */
    int scores_idx = interpreter->outputs()[0];
    TfLiteIntArray *rgr_dim = interpreter->tensor(scores_idx)->dims;
    int r0 = rgr_dim->data[0];
    int r1 = rgr_dim->data[1];
    int r2 = rgr_dim->data[2];
    DBG_LOGI ("scores dim    : (%dx%dx%d)\n", r0, r1, r2);

    /* classificators dimention */
    int bboxes_idx = interpreter->outputs()[1];
    TfLiteIntArray *cls_dim = interpreter->tensor(bboxes_idx)->dims;
    int c0 = cls_dim->data[0];
    int c1 = cls_dim->data[1];
    int c2 = cls_dim->data[2];
    DBG_LOGI ("bboxes dim    : (%dx%dx%d)\n", c0, c1, c2);

    int anchor_num = create_blazeface_anchors (s_img_w, s_img_h);
    DBG_LOGI ("anchors num   : %d\n", anchor_num);

    if (anchor_num != c1)
    {
        DBG_LOGE ("ERR: %s(%d): anchor num doesn't much (%d != %d)\n",
                    __FILE__, __LINE__, anchor_num, c1);
        return -1;
    }

    return 0;
}

void *
get_blazeface_input_buf (int *w, int *h)
{
    *w = s_img_w;
    *h = s_img_h;
    return in_ptr;
}


static float *
get_bbox_ptr (int anchor_idx)
{
    int idx = 16 * anchor_idx;
    return &bboxes_ptr[idx];
}


static int
decode_bounds (std::list<face_t> &face_list, float score_thresh)
{
    face_t face_item;

    int i = 0;
    for (auto itr = s_anchors.begin(); itr != s_anchors.end(); i ++, itr ++)
    {
        fvec2 anchor = *itr;
        float score0 = scores_ptr[i];
        float score = 1.0f / (1.0f + exp(-score0));

        if (score > score_thresh)
        {
            float *p = get_bbox_ptr (i);

            /* boundary box */
            float sx = p[0];
            float sy = p[1];
            float w  = p[2];
            float h  = p[3];

            float cx = sx + anchor.x;
            float cy = sy + anchor.y;

            cx /= (float)s_img_w;
            cy /= (float)s_img_h;
            w  /= (float)s_img_w;
            h  /= (float)s_img_h;

            fvec2 topleft, btmright;
            topleft.x  = cx - w * 0.5f;
            topleft.y  = cy - h * 0.5f;
            btmright.x = cx + w * 0.5f;
            btmright.y = cy + h * 0.5f;

            face_item.score    = score;
            face_item.topleft  = topleft;
            face_item.btmright = btmright;

            /* landmark positions (6 keys) */
            for (int j = 0; j < kFaceKeyNum; j ++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x;
                ly += anchor.y;
                lx /= (float)s_img_w;
                ly /= (float)s_img_h;

                face_item.keys[j].x = lx;
                face_item.keys[j].y = ly;
            }

            face_list.push_back (face_item);
        }
    }
    return 0;
}


/* -------------------------------------------------- *
 *  Apply NonMaxSuppression:
 *      https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/image_ops.ts
 * -------------------------------------------------- */
static float
calc_intersection_over_union (face_t &face0, face_t &face1)
{
    float sx0 = face0.topleft.x;
    float sy0 = face0.topleft.y;
    float ex0 = face0.btmright.x;
    float ey0 = face0.btmright.y;
    float sx1 = face1.topleft.x;
    float sy1 = face1.topleft.y;
    float ex1 = face1.btmright.x;
    float ey1 = face1.btmright.y;
    
    float xmin0 = min (sx0, ex0);
    float ymin0 = min (sy0, ey0);
    float xmax0 = max (sx0, ex0);
    float ymax0 = max (sy0, ey0);
    float xmin1 = min (sx1, ex1);
    float ymin1 = min (sy1, ey1);
    float xmax1 = max (sx1, ex1);
    float ymax1 = max (sy1, ey1);
    
    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    if (area0 <= 0 || area1 <= 0)
        return 0.0f;

    float intersect_xmin = max (xmin0, xmin1);
    float intersect_ymin = max (ymin0, ymin1);
    float intersect_xmax = min (xmax0, xmax1);
    float intersect_ymax = min (ymax0, ymax1);

    float intersect_area = max (intersect_ymax - intersect_ymin, 0.0f) *
                           max (intersect_xmax - intersect_xmin, 0.0f);
    
    return intersect_area / (area0 + area1 - intersect_area);
}

static bool
compare (face_t &v1, face_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}

static int
non_max_suppression (std::list<face_t> &face_list, std::list<face_t> &face_sel_list, float iou_thresh)
{
    face_list.sort (compare);

    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        face_t face_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_sel = face_sel_list.rbegin(); itr_sel != face_sel_list.rend(); itr_sel ++)
        {
            face_t face_sel = *itr_sel;

            float iou = calc_intersection_over_union (face_candidate, face_sel);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            face_sel_list.push_back(face_candidate);
            if (face_sel_list.size() >= MAX_FACE_NUM)
                break;
        }
    }

    return 0;
}

static void
pack_face_result (blazeface_result_t *face_result, std::list<face_t> &face_list)
{
    int num_faces = 0;
    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        face_t face = *itr;
        memcpy (&face_result->faces[num_faces], &face, sizeof (face));
        num_faces ++;
        face_result->num = num_faces;

        if (num_faces >= MAX_FACE_NUM)
            break;
    }
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_blazeface (blazeface_result_t *face_result)
{
    if (interpreter->Invoke() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* decode boundary box and landmark keypoints */
    float score_thresh = 0.5f;//0.75f;
    std::list<face_t> face_list;

    decode_bounds (face_list, score_thresh);


#if 1 /* USE NMS */
    float iou_thresh = 0.3f;
    std::list<face_t> face_nms_list;

    non_max_suppression (face_list, face_nms_list, iou_thresh);
    pack_face_result (face_result, face_nms_list);
#else
    pack_face_result (face_result, face_list);
#endif

    return 0;
}

