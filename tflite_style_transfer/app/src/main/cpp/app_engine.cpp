/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <cstdio>
#include <GLES2/gl2.h>
#include "app_engine.h"
#include "util_debug.h"
#include "util_asset.h"
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"

#define MAX_BUF_COUNT 4


AppEngine::AppEngine (android_app* app)
    : m_app(app),
      m_cameraGranted(false),
      m_camera(nullptr),
      m_img_reader(nullptr)
{
    memset (&glctx, 0, sizeof (glctx));
}

AppEngine::~AppEngine() 
{
    DeleteCamera();
}


/* ---------------------------------------------------------------------------- *
 *  Interfaces to android application framework
 * ---------------------------------------------------------------------------- */
void
AppEngine::OnAppInitWindow (void)
{
    InitGLES();
    InitCamera();
}

void
AppEngine::OnAppTermWindow (void)
{
    DeleteCamera();
    TerminateGLES();
}

struct android_app *
AppEngine::AndroidApp (void) const
{
    return m_app;
}


/* ---------------------------------------------------------------------------- *
 *  OpenGLES Render Functions
 * ---------------------------------------------------------------------------- */
void
AppEngine::LoadInputTexture (input_tex_t *tex, char *fname)
{
    int32_t dsp_w = glctx.disp_w;
    int32_t dsp_h = glctx.disp_h;
    int32_t img_w, img_h, drw_x, drw_y, drw_w, drw_h;
    uint8_t *img_buf;
    GLuint texid;

    img_buf = asset_read_image (AndroidApp()->activity->assetManager, fname, &img_w, &img_h);
    texid = create_2d_texture ((void *)img_buf, img_w, img_h);
    asset_free_image (img_buf);

    AdjustTexture (dsp_w, dsp_h, img_w, img_h, &drw_x, &drw_y, &drw_w, &drw_h);

    tex->w = img_w;
    tex->h = img_h;
    tex->texid = texid;
    tex->draw_x = drw_x;
    tex->draw_y = drw_y;
    tex->draw_w = drw_w;
    tex->draw_h = drw_h;
}


/* Adjust the texture size to fit the window size
 *
 *                      Portrait
 *     Landscape        +------+
 *     +-+------+-+     +------+
 *     | |      | |     |      |
 *     | |      | |     |      |
 *     +-+------+-+     +------+
 *                      +------+
 */
void
AppEngine::AdjustTexture (int win_w, int win_h, int texw, int texh,
                          int *dx, int *dy, int *dw, int *dh)
{
    float win_aspect = (float)win_w / (float)win_h;
    float tex_aspect = (float)texw  / (float)texh;
    float scale;
    float scaled_w, scaled_h;
    float offset_x, offset_y;

    if (win_aspect > tex_aspect)
    {
        scale = (float)win_h / (float)texh;
        scaled_w = scale * texw;
        scaled_h = scale * texh;
        offset_x = (win_w - scaled_w) * 0.5f;
        offset_y = 0;
    }
    else
    {
        scale = (float)win_w / (float)texw;
        scaled_w = scale * texw;
        scaled_h = scale * texh;
        offset_x = 0;
        offset_y = (win_h - scaled_h) * 0.5f;
    }

    *dx = (int)offset_x;
    *dy = (int)offset_y;
    *dw = (int)scaled_w;
    *dh = (int)scaled_h;
}


/* resize image to DNN network input size and convert to fp32. */
void
AppEngine::FeedInputImage (int is_predict, int texid, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32;
    unsigned char *buf_ui8, *pui8;

    if (is_predict)
        buf_fp32 = (float *)get_style_predict_input_buf (&w, &h);
    else
        buf_fp32 = (float *)get_style_transfer_content_input_buf (&w, &h);

    pui8 = buf_ui8 = (unsigned char *)malloc(w * h * 4);

    draw_2d_texture (texid, 0, win_h - h, w, h, 1);

    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

#if 0
    /* convert UI8 [0, 255] ==> FP32 [-1, 1] */
    float mean = 128.0f;
    float std  = 128.0f;
#else
    /* convert UI8 [0, 255] ==> FP32 [ 0, 1] */
    float mean =   0.0f;
    float std  = 255.0f;
#endif
    for (y = 0; y < h; y ++)
    {
        for (x = 0; x < w; x ++)
        {
            int r = *buf_ui8 ++;
            int g = *buf_ui8 ++;
            int b = *buf_ui8 ++;
            buf_ui8 ++;          /* skip alpha */
            *buf_fp32 ++ = (float)(r - mean) / std;
            *buf_fp32 ++ = (float)(g - mean) / std;
            *buf_fp32 ++ = (float)(b - mean) / std;
        }
    }

    free (pui8);
    return;
}


void
AppEngine::FeedBlendStyle (style_predict_t *style0, style_predict_t *style1, float ratio)
{
    int size;
    float *s0 = (float *)style0->param;
    float *s1 = (float *)style1->param;
    float *d  = (float *)get_style_transfer_style_input_buf (&size);

    if (style0->size != size || style1->size != size)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return;
    }
    
    for (int i = 0; i < size; i ++)
    {
        float src0 = *s0;
        float src1 = *s1;
        float src  = (ratio * src1) + ((1.0f - ratio) * src0);

        *d ++ = src;
        s0 ++;
        s1 ++;
    }
}


/* copy predicted style parameter to local buffer */
void
AppEngine::StoreStylePredictParam (style_predict_t *style)
{
    int size = style->size;
    float *param = (float *)style->param;
    float *store_param = (float *)calloc (1, size * sizeof(float));

    style->param = store_param;

    while (size --> 0)
    {
        *store_param ++ = *param ++;
    }
}


/* upload style transfered image to OpenGLES texture */
static int
update_style_transfered_texture (style_transfer_t *transfer)
{
    static int s_texid = 0;
    static uint8_t *s_texbuf = NULL;
    int img_w = transfer->w;
    int img_h = transfer->h;

    if (s_texid == 0)
    {
        s_texbuf = (uint8_t *)calloc (1, img_w * img_h * 4);
        s_texid  = create_2d_texture (s_texbuf, img_w, img_h);
    }

    uint8_t *d = s_texbuf;
    float   *s = (float *)transfer->img;

    for (int y = 0; y < img_h; y ++)
    {
        for (int x = 0; x < img_w; x ++)
        {
            float r = *s ++;
            float g = *s ++;
            float b = *s ++;
            *d ++ = (uint8_t)(r * 255);
            *d ++ = (uint8_t)(g * 255);
            *d ++ = (uint8_t)(b * 255);
            *d ++ = 0xFF;
        }
    }

    glBindTexture (GL_TEXTURE_2D, s_texid);
    glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, img_w, img_h, GL_RGBA, GL_UNSIGNED_BYTE, s_texbuf);

    return s_texid;
}


void
AppEngine::DrawTFLiteConfigInfo ()
{
    char strbuf[512];
    float col_pink[]  = {1.0f, 0.0f, 1.0f, 0.5f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float *col_bg = col_pink;

    if (glctx.tex_camera_valid)
    {
        sprintf (strbuf, "CAMERA ENABLED");
    }
    else
    {
        sprintf (strbuf, "CAMERA DISABLED");
    }
    draw_dbgstr_ex (strbuf, glctx.disp_w - 250, 0, 1.0f, col_white, col_bg);

#if defined (USE_GPU_DELEGATEV2)
    sprintf (strbuf, "GPU_DELEGATEV2: ON ");
#else
    sprintf (strbuf, "GPU_DELEGATEV2: OFF");
#endif
    draw_dbgstr_ex (strbuf, glctx.disp_w - 250, 24, 1.0f, col_white, col_bg);

#if defined (USE_QUANT_TFLITE_MODEL)
    sprintf (strbuf, "MODEL_INTQUANT: ON ");
    draw_dbgstr_ex (strbuf, glctx.disp_w - 250, 48, 1.0f, col_white, col_bg);
#endif
}


void 
AppEngine::RenderFrame ()
{
    input_tex_t *input_tex;

    if (glctx.tex_camera_valid)
        input_tex = &glctx.tex_camera;
    else
        input_tex = &glctx.tex_static;

    GLuint texid = input_tex->texid;
    int draw_x = input_tex->draw_x;
    int draw_y = input_tex->draw_y;
    int draw_w = input_tex->draw_w;
    int draw_h = input_tex->draw_h;
    int win_w  = glctx.disp_w;
    int win_h  = glctx.disp_h;
    int style_texid = glctx.tex_style.texid;
    static double ttime[10] = {0}, interval, invoke_ms;

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    int count = glctx.frame_count;

    /* --------------------------------------- *
     *  Style prediction
     * --------------------------------------- */
    if (count == 0)
    {
        glClear (GL_COLOR_BUFFER_BIT);

        /* predict style of original image */
        glClear (GL_COLOR_BUFFER_BIT);
        FeedInputImage (1, texid, win_w, win_h);
        invoke_style_predict (&m_style_predict[0]);
        StoreStylePredictParam (&m_style_predict[0]);

        /* predict style of target image */
        glClear (GL_COLOR_BUFFER_BIT);
        FeedInputImage (1, style_texid, win_w, win_h);
        invoke_style_predict (&m_style_predict[1]);
        StoreStylePredictParam (&m_style_predict[1]);

        m_style_ratio = -0.1f;
    }

    /* --------------------------------------- *
     *  Style transfer
     * --------------------------------------- */
    {
        style_transfer_t style_transfered = {};
        float style_ratio = m_style_ratio;

        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);

#if 0
        /* 
         *  update style parameter blend ratio.
         *      0.0: apply 100[%] style of original image.
         *      1.0: apply 100[%] style of target image.
         */
        style_ratio += 0.1f;
        if (style_ratio > 1.01f)
            style_ratio = -0.1f;
#else
        style_ratio = 1.0f;
#endif
        m_style_ratio = style_ratio;

        /* feed style parameter and original image */
        FeedBlendStyle (&m_style_predict[0], &m_style_predict[1], style_ratio);
        FeedInputImage (0, texid, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_style_transfer (&style_transfered);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* visualize the style transform results. */
        glClear (GL_COLOR_BUFFER_BIT);
        int transfered_texid = update_style_transfered_texture (&style_transfered);

        if (style_ratio < 0.0f)     /* render original content image */
            draw_2d_texture (texid,  draw_x, draw_y, draw_w, draw_h, 0);
        else                        /* render style transformed image */
            draw_2d_texture (transfered_texid,  draw_x, draw_y, draw_w, draw_h, 0);

        /* render the target style image */
        {
            float col_black[] = {1.0f, 1.0f, 1.0f, 1.0f};
            draw_2d_texture (style_texid,  win_w - 200, 0, 200, 200, 0);
            draw_2d_rect (win_w - 200, 0, 200, 200, col_black, 2.0f);
        }

        DrawTFLiteConfigInfo ();

        /* renderer info */
        draw_dbgstr (glctx.str_glverstion, 10, 0);
        draw_dbgstr (glctx.str_glvendor,   10, 22);
        draw_dbgstr (glctx.str_glrender,   10, 44);

        draw_pmeter (0, 100);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]\nstyle_ratio=%.1f", 
                                interval, invoke_ms, style_ratio);
        draw_dbgstr (strbuf, 10, 80);

        egl_swap();
    }
    glctx.frame_count ++;
}


void
AppEngine::UpdateCameraTexture ()
{
    input_tex_t *input_tex = &glctx.tex_camera;
    GLuint texid = input_tex->texid;
    int    cap_w, cap_h;
    void   *cap_buf;

    GetAppEngine()->AcquireCameraFrame (&cap_buf, &cap_w, &cap_h);

    if (cap_buf == NULL)
        return;

    if (texid == 0)
    {
        int32_t dsp_w = glctx.disp_w;
        int32_t dsp_h = glctx.disp_h;
        int32_t drw_x, drw_y, drw_w, drw_h;

        texid = create_2d_texture (cap_buf, cap_w, cap_h);

        AdjustTexture (dsp_w, dsp_h, cap_w, cap_h, &drw_x, &drw_y, &drw_w, &drw_h);

        input_tex->w      = cap_w;
        input_tex->h      = cap_h;
        input_tex->texid  = texid;
        input_tex->draw_x = drw_x;
        input_tex->draw_y = drw_y;
        input_tex->draw_w = drw_w;
        input_tex->draw_h = drw_h;
    }
    else
    {
        glBindTexture (GL_TEXTURE_2D, texid);
        glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, cap_w, cap_h, GL_RGBA, GL_UNSIGNED_BYTE, cap_buf);
    }

    glctx.tex_camera_valid = true;
}


void 
AppEngine::InitGLES (void)
{
    int ret;

    egl_init_with_window_surface (2, m_app->window, 8, 0, 0);

    glctx.str_glverstion = (char *)glGetString (GL_VERSION);
    glctx.str_glvendor   = (char *)glGetString (GL_VENDOR);
    glctx.str_glrender   = (char *)glGetString (GL_RENDERER);

    int w, h;
    egl_get_current_surface_dimension (&w, &h);

    init_2d_renderer (w, h);
    init_pmeter (w, h, h - 100);
    init_dbgstr (w, h);

    asset_read_file (m_app->activity->assetManager,
                    (char *)STYLE_PREDICT_MODEL_PATH, m_style_predict_tflite_model_buf);

    asset_read_file (m_app->activity->assetManager,
                    (char *)STYLE_TRANSFER_MODEL_PATH, m_style_transfer_tflite_model_buf);

    ret = init_tflite_style_transfer (
        (const char *)m_style_predict_tflite_model_buf.data(), m_style_predict_tflite_model_buf.size(),
        (const char *)m_style_transfer_tflite_model_buf.data(), m_style_transfer_tflite_model_buf.size());

    glctx.disp_w = w;
    glctx.disp_h = h;
    LoadInputTexture (&glctx.tex_static, (char *)"pakutaso_famicom.jpg");
    LoadInputTexture (&glctx.tex_style,  (char *)"munch_scream.jpg");

    glctx.initdone = 1;
}


void
AppEngine::TerminateGLES (void)
{
    egl_terminate ();
}


void
AppEngine::UpdateFrame (void)
{
    if (glctx.initdone == 0)
        return;

    if (m_cameraGranted)
    {
        UpdateCameraTexture();
    }

    RenderFrame();
}


/* ---------------------------------------------------------------------------- *
 *  Manage NDKCamera Functions
 * ---------------------------------------------------------------------------- */
void 
AppEngine::InitCamera (void)
{
    // Not permitted to use camera yet, ask(again) and defer other events
    if (!m_cameraGranted)
    {
        RequestCameraPermission();
        return;
    }

    CreateCamera();
}


void
AppEngine::DeleteCamera(void)
{
    if (m_camera)
    {
        delete m_camera;
        m_camera = nullptr;
    }

    if (m_img_reader)
    {
        AImageReader_delete(m_img_reader);
        m_img_reader = nullptr;
    }
}


void 
AppEngine::CreateCamera(void) 
{
    m_camera = new NDKCamera();
    ASSERT (m_camera, "Failed to Create CameraObject");

    int32_t cam_w, cam_h, cam_fmt;
    m_camera->MatchCaptureSizeRequest (&cam_w, &cam_h, &cam_fmt);

    media_status_t status;
    status = AImageReader_new (cam_w, cam_h, cam_fmt, MAX_BUF_COUNT, &m_img_reader);
    ASSERT (status == AMEDIA_OK, "Failed to create AImageReader");

    ANativeWindow *nativeWindow;
    status = AImageReader_getWindow (m_img_reader, &nativeWindow);
    ASSERT (status == AMEDIA_OK, "Could not get ANativeWindow");

    m_camera->CreateSession (nativeWindow);

    m_camera->StartPreview (true);
}





/**
 * Helper function for YUV_420 to RGB conversion. Courtesy of Tensorflow
 * ImageClassifier Sample:
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/jni/yuv2rgb.cc
 * The difference is that here we have to swap UV plane when calling it.
 */
#ifndef MAX
#define MAX(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;      \
  })
#define MIN(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;      \
  })
#endif

// This value is 2 ^ 18 - 1, and is used to clamp the RGB values before their
// ranges are normalized to eight bits.
static const int kMaxChannelValue = 262143;

static inline uint32_t
YUV2RGB(int nY, int nU, int nV)
{
    nY -= 16;
    nU -= 128;
    nV -= 128;
    if (nY < 0) nY = 0;

    // This is the floating point equivalent. We do the conversion in integer
    // because some Android devices do not have floating point in hardware.
    // nR = (int)(1.164 * nY + 1.596 * nV);
    // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
    // nB = (int)(1.164 * nY + 2.018 * nU);

    int nR = (int)(1192 * nY + 1634 * nV);
    int nG = (int)(1192 * nY - 833 * nV - 400 * nU);
    int nB = (int)(1192 * nY + 2066 * nU);

    nR = MIN(kMaxChannelValue, MAX(0, nR));
    nG = MIN(kMaxChannelValue, MAX(0, nG));
    nB = MIN(kMaxChannelValue, MAX(0, nB));

    nR = (nR >> 10) & 0xff;
    nG = (nG >> 10) & 0xff;
    nB = (nB >> 10) & 0xff;

    return 0xff000000 | (nR << 16) | (nG << 8) | nB;
}

void
AppEngine::AcquireCameraFrame (void **cap_buf, int *cap_w, int *cap_h)
{
    static uint32_t *s_cap_buf = NULL;

    *cap_buf = NULL;
    *cap_w   = 0;
    *cap_h   = 0;

    if (!m_img_reader) 
        return;

    AImage *image;
    media_status_t status = AImageReader_acquireLatestImage(m_img_reader, &image);
    if (status != AMEDIA_OK) 
    {
        return;
    }

    int32_t img_w, img_h;
    int32_t yStride, uvStride;
    uint8_t *yPixel, *uPixel, *vPixel;
    int32_t yLen, uLen, vLen;
    int32_t uvPixelStride;
    AImageCropRect srcRect;

    AImage_getWidth           (image, &img_w);
    AImage_getHeight          (image, &img_h);
    AImage_getPlaneRowStride  (image, 0, &yStride);
    AImage_getPlaneRowStride  (image, 1, &uvStride);
    AImage_getPlaneData       (image, 0, &yPixel, &yLen);
    AImage_getPlaneData       (image, 1, &vPixel, &vLen);
    AImage_getPlaneData       (image, 2, &uPixel, &uLen);
    AImage_getPlanePixelStride(image, 1, &uvPixelStride);
    AImage_getCropRect        (image, &srcRect);

    int32_t width  = srcRect.right  - srcRect.left;
    int32_t height = srcRect.bottom - srcRect.top ;
    int32_t x, y;

    if (s_cap_buf == NULL)
    {
        s_cap_buf = (uint32_t *)calloc (4, img_w * img_h);
    }

    uint32_t *out = s_cap_buf;
    for (y = 0; y < height; y++) 
    {
        const uint8_t *pY = yPixel + yStride * (y + srcRect.top) + srcRect.left;

        int32_t uv_row_start = uvStride * ((y + srcRect.top) >> 1);
        const uint8_t *pU = uPixel + uv_row_start + (srcRect.left >> 1);
        const uint8_t *pV = vPixel + uv_row_start + (srcRect.left >> 1);

        for (x = 0; x < width; x++) 
        {
            const int32_t uv_offset = (x >> 1) * uvPixelStride;
            out[x] = YUV2RGB (pY[x], pU[uv_offset], pV[uv_offset]);
        }
        out += img_w;
    }

    AImage_delete (image);

    *cap_buf = s_cap_buf;
    *cap_w   = img_w;
    *cap_h   = img_h;
}


/* --------------------------------------------------------------------------- *
 * Initiate a Camera Run-time usage request to Java side implementation
 *  [The request result will be passed back in function notifyCameraPermission()]
 * --------------------------------------------------------------------------- */
void
AppEngine::RequestCameraPermission()
{
    if (!AndroidApp())
        return;

    ANativeActivity *activity = AndroidApp()->activity;

    JNIEnv *env;
    activity->vm->GetEnv ((void**)&env, JNI_VERSION_1_6);
    activity->vm->AttachCurrentThread (&env, NULL);

    jobject activityObj = env->NewGlobalRef (activity->clazz);
    jclass clz = env->GetObjectClass (activityObj);

    env->CallVoidMethod (activityObj, env->GetMethodID (clz, "RequestCamera", "()V"));
    env->DeleteGlobalRef (activityObj);

    activity->vm->DetachCurrentThread();
}

void
AppEngine::OnCameraPermission (jboolean granted)
{
    m_cameraGranted = (granted != JNI_FALSE);

    if (m_cameraGranted)
    {
        InitCamera();
    }
}


extern "C" JNIEXPORT void JNICALL
Java_com_glesapp_glesapp_GLESAppNativeActivity_notifyCameraPermission (
                            JNIEnv *env, jclass type, jboolean permission)
{
    std::thread permissionHandler (&AppEngine::OnCameraPermission, GetAppEngine(), permission);
    permissionHandler.detach();
}

