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


void
AppEngine::FeedInputImageUI8 (int texid, int win_w, int win_h)
{
    int w, h;
    uint8_t *buf_u8 = (uint8_t *)get_deeplab_input_buf (&w, &h);

    draw_2d_texture (texid, 0, win_h - h, w, h, 1);

#if 0 /* if your platform supports glReadPixles(GL_RGB), use this code. */
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glReadPixels (0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, buf);
#else /* if your platform supports only glReadPixels(GL_RGBA), try this code. */
    {
        int x, y;
        unsigned char *bufRGBA = (unsigned char *)malloc (w * h * 4);
        unsigned char *pRGBA = bufRGBA;
        glPixelStorei (GL_PACK_ALIGNMENT, 4);
        glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, bufRGBA);

        for (y = 0; y < h; y ++)
        {
            for (x = 0; x < w; x ++)
            {
                int r = *pRGBA ++;
                int g = *pRGBA ++;
                int b = *pRGBA ++;
                pRGBA ++;          /* skip alpha */

                *buf_u8 ++ = r;
                *buf_u8 ++ = g;
                *buf_u8 ++ = b;
            }
        }
        free (bufRGBA);
    }
#endif
}

/* resize image to DNN network input size and convert to fp32. */
void
AppEngine::FeedInputImageFP32 (int texid, int win_w, int win_h)
{
    int w, h;
    float *buf_fp32 = (float *)get_deeplab_input_buf (&w, &h);

    draw_2d_texture (texid, 0, win_h - h, w, h, 1);

    int x, y;
    unsigned char *bufRGBA = (unsigned char *)malloc (w * h * 4);
    unsigned char *pRGBA = bufRGBA;
    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, bufRGBA);

    /* convert UI8 [0, 255] ==> FP32 [-1, 1] */
    float mean = 128.0f;
    float std  = 128.0f;
    for (y = 0; y < h; y ++)
    {
        for (x = 0; x < w; x ++)
        {
            int r = *pRGBA ++;
            int g = *pRGBA ++;
            int b = *pRGBA ++;
            pRGBA ++;          /* skip alpha */
            *buf_fp32 ++ = (float)(r - mean) / std;
            *buf_fp32 ++ = (float)(g - mean) / std;
            *buf_fp32 ++ = (float)(b - mean) / std;
        }
    }
    free (bufRGBA);
}


void
AppEngine::FeedInputImage (int texid, int win_w, int win_h)
{
    int type = get_deeplab_input_type ();
    if (type)
        FeedInputImageUI8  (texid, win_w, win_h);
    else
        FeedInputImageFP32 (texid, win_w, win_h);
}


void
render_deeplab_result (int ofstx, int ofsty, int draw_w, int draw_h, deeplab_result_t *deeplab_ret)
{
    float *segmap = deeplab_ret->segmentmap;
    int segmap_w  = deeplab_ret->segmentmap_dims[0];
    int segmap_h  = deeplab_ret->segmentmap_dims[1];
    int segmap_c  = deeplab_ret->segmentmap_dims[2];
    int x, y, c;
    unsigned int imgbuf[segmap_h][segmap_w];

    /* find the most confident class for each pixel. */
    for (y = 0; y < segmap_h; y ++)
    {
        for (x = 0; x < segmap_w; x ++)
        {
            int max_id;
            float conf_max = 0;
            for (c = 0; c < 21; c ++)
            {
                float confidence = segmap[(y * segmap_w * segmap_c)+ (x * segmap_c) + c];
                if (c == 0 || confidence > conf_max)
                {
                    conf_max = confidence;
                    max_id = c;
                }
            }
            float *col = get_deeplab_class_color (max_id);
            unsigned char r = ((int)(col[0] * 255)) & 0xff;
            unsigned char g = ((int)(col[1] * 255)) & 0xff;
            unsigned char b = ((int)(col[2] * 255)) & 0xff;
            unsigned char a = ((int)(col[3] * 255)) & 0xff;
            imgbuf[y][x] = (a << 24) | (b << 16) | (g << 8) | (r);
        }
    }
    
    GLuint texid;
    glGenTextures (1, &texid );
    glBindTexture (GL_TEXTURE_2D, texid);

    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei (GL_UNPACK_ALIGNMENT, 4);

    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA,
        segmap_w, segmap_h, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, imgbuf);

    draw_2d_texture (texid, ofstx, ofsty, draw_w, draw_h, 0);

    /* class name */
    for (c = 0; c < 21; c ++)
    {
        float col_str[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float *col = get_deeplab_class_color (c);
        char *name = get_deeplab_class_name (c);
        char buf[512];
        sprintf (buf, "%2d:%s", c, name);
        draw_dbgstr_ex (buf, ofstx, ofsty + c * 22 * 0.7, 0.7f, col_str, col);
    }

    glDeleteTextures (1, &texid);
}

void
render_deeplab_heatmap (int ofstx, int ofsty, int draw_w, int draw_h, deeplab_result_t *deeplab_ret)
{
    float *segmap = deeplab_ret->segmentmap;
    int segmap_w  = deeplab_ret->segmentmap_dims[0];
    int segmap_h  = deeplab_ret->segmentmap_dims[1];
    int segmap_c  = deeplab_ret->segmentmap_dims[2];
    int x, y;
    unsigned char imgbuf[segmap_h][segmap_w];
    static int s_count = 0;
    int key_id = (s_count /10)% 21;
    s_count ++;
    float conf_min, conf_max;


#if 1
    conf_min =  0.0f;
    conf_max = 50.0f;
#else
    conf_min =  FLT_MAX;
    conf_max = -FLT_MAX;
    for (y = 0; y < segmap_h; y ++)
    {
        for (x = 0; x < segmap_w; x ++)
        {
            float confidence = segmap[(y * segmap_w * segmap_c)+ (x * segmap_c) + key_id];
            if (confidence < conf_min) conf_min = confidence;
            if (confidence > conf_max) conf_max = confidence;
        }
    }
#endif

    for (y = 0; y < segmap_h; y ++)
    {
        for (x = 0; x < segmap_w; x ++)
        {
            float confidence = segmap[(y * segmap_w * segmap_c)+ (x * segmap_c) + key_id];
            confidence = (confidence - conf_min) / (conf_max - conf_min);
            if (confidence < 0.0f) confidence = 0.0f;
            if (confidence > 1.0f) confidence = 1.0f;
            imgbuf[y][x] = confidence * 255;
        }
    }
    
    GLuint texid;
    glGenTextures (1, &texid );
    glBindTexture (GL_TEXTURE_2D, texid);

    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei (GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D (GL_TEXTURE_2D, 0, GL_LUMINANCE,
        segmap_w, segmap_h, 0, GL_LUMINANCE,
        GL_UNSIGNED_BYTE, imgbuf);

    draw_2d_colormap (texid, ofstx, ofsty, draw_w, draw_h, 0.8f, 0);

    glDeleteTextures (1, &texid);

    {
        char strbuf[128];
        sprintf (strbuf, "%2d (%f, %f) %s\n", key_id, 
            conf_min, conf_max, get_deeplab_class_name (key_id));
        draw_dbgstr (strbuf, ofstx + 5, 5);
    }
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
    static double ttime[10] = {0}, interval, invoke_ms;

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    int count = glctx.frame_count;
    {
        deeplab_result_t deeplab_result;
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);

        FeedInputImage (texid, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_deeplab (&deeplab_result);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* visualize the object detection results. */
        glClear (GL_COLOR_BUFFER_BIT);
        draw_2d_texture (texid,  draw_x, draw_y, draw_w, draw_h, 0);
        render_deeplab_result (draw_x, draw_y, draw_w, draw_h, &deeplab_result);

        DrawTFLiteConfigInfo ();

        /* renderer info */
        draw_dbgstr (glctx.str_glverstion, 10, 0);
        draw_dbgstr (glctx.str_glvendor,   10, 22);
        draw_dbgstr (glctx.str_glrender,   10, 44);

        draw_pmeter (0, 100);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]", interval, invoke_ms);
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
                    (char *)DEEPLAB_MODEL_PATH, m_detect_tflite_model_buf);

    ret = init_tflite_deeplab (
        (const char *)m_detect_tflite_model_buf.data(), m_detect_tflite_model_buf.size());

    glctx.disp_w = w;
    glctx.disp_h = h;
    LoadInputTexture (&glctx.tex_static, (char *)"ride_horse.jpg");

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

