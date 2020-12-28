/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <cstdio>
#include <GLES2/gl2.h>
#include "util_debug.h"
#include "util_asset.h"
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "tflite_blazeface.h"
#include "app_engine.h"
#include "render_imgui.h"

#define UNUSED(x) (void)(x)





/* resize image to DNN network input size and convert to fp32. */
void
AppEngine::feed_blazeface_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_blazeface_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    draw_2d_texture_ex (srctex, 0, win_h - h, w, h, 1);

    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

    /* convert UI8 [0, 255] ==> FP32 [-1, 1] */
    float mean = 128.0f;
    float std  = 128.0f;
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

    return;
}


void
AppEngine::render_detect_region (int ofstx, int ofsty, int texw, int texh,
                                 blazeface_result_t *detection, imgui_data_t *imgui_data)
{
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float *col_frame = imgui_data->frame_color;

    for (int i = 0; i < detection->num; i ++)
    {
        face_t *face = &(detection->faces[i]);
        float x1 = face->topleft.x  * texw + ofstx;
        float y1 = face->topleft.y  * texh + ofsty;
        float x2 = face->btmright.x * texw + ofstx;
        float y2 = face->btmright.y * texh + ofsty;
        float score = face->score;

        /* rectangle region */
        draw_2d_rect (x1, y1, x2-x1, y2-y1, col_frame, 2.0f);

        /* detect score */
        char buf[512];
        sprintf (buf, "%d", (int)(score * 100));
        draw_dbgstr_ex (buf, x1, y1, 1.0f, col_white, col_frame);

        /* key points */
        for (int j = 0; j < kFaceKeyNum; j ++)
        {
            float x = face->keys[j].x * texw + ofstx;
            float y = face->keys[j].y * texh + ofsty;

            int r = 4;
            draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_frame);
        }
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
AppEngine::adjust_texture (int win_w, int win_h, int texw, int texh, 
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

#if defined (USE_IMGUI)
void
AppEngine::mousemove_cb (int x, int y)
{
    imgui_mousemove (x, y);
}

void
AppEngine::button_cb (int button, int state, int x, int y)
{
    imgui_mousebutton (button, state, x, y);
}

void
AppEngine::keyboard_cb (int key, int state, int x, int y)
{
}
#endif

void
AppEngine::setup_imgui (int win_w, int win_h, imgui_data_t *imgui_data)
{
#if defined (USE_IMGUI)
    //egl_set_motion_func (mousemove_cb);
    //egl_set_button_func (button_cb);
    //egl_set_key_func    (keyboard_cb);

    init_imgui (win_w, win_h);
#endif

    imgui_data->frame_color[0] = 1.0f;
    imgui_data->frame_color[1] = 0.0f;
    imgui_data->frame_color[2] = 0.0f;
    imgui_data->frame_color[3] = 1.0f;
}


void 
AppEngine::RenderFrame ()
{
    texture_2d_t captex;

    if (glctx.tex_camera_valid)
        captex = glctx.tex_camera;
    else
        captex = glctx.tex_static;

    int win_w  = glctx.disp_w;
    int win_h  = glctx.disp_h;
    static double ttime[10] = {0}, interval, invoke_ms;

    int draw_x, draw_y, draw_w, draw_h;
	int texw = captex.width;
	int texh = captex.height;
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h);

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
    int count = glctx.frame_count;
    {
        blazeface_result_t face_ret = {0};
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);

        /* --------------------------------------- *
         *  face detection
         * --------------------------------------- */
        feed_blazeface_image (&captex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_blazeface (&face_ret, &imgui_data.blazeface_config);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  render scene
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT);

        /* visualize the face detection results. */
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);
        render_detect_region (draw_x, draw_y, draw_w, draw_h, &face_ret, &imgui_data);

        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        DrawTFLiteConfigInfo ();

        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]", interval, invoke_ms);
        draw_dbgstr (strbuf, 10, 10);

        /* renderer info */
		int y = 10 + 22 * 2;
        draw_dbgstr (glctx.str_glverstion, 10, y); y += 22;
        draw_dbgstr (glctx.str_glvendor,   10, y); y += 22;
        draw_dbgstr (glctx.str_glrender,   10, y); y += 22;

#if defined (USE_IMGUI)
        invoke_imgui (&imgui_data);
#endif
        egl_swap();
    }
    glctx.frame_count ++;
}


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
AppEngine::LoadInputTexture (texture_2d_t *tex, char *fname)
{
    int32_t img_w, img_h;
    uint8_t *img_buf = asset_read_image (AndroidApp()->activity->assetManager, fname, &img_w, &img_h);

    create_2d_texture_ex (tex, img_buf, img_w, img_h, pixfmt_fourcc('R', 'G', 'B', 'A'));
    asset_free_image (img_buf);
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
                    (char *)BLAZEFACE_MODEL_PATH, m_tflite_model_buf);

    ret = init_tflite_blazeface (
        (const char *)m_tflite_model_buf.data(), m_tflite_model_buf.size(),
        &imgui_data.blazeface_config);

    setup_imgui (w, h, &imgui_data);

    glctx.disp_w = w;
    glctx.disp_h = h;
    LoadInputTexture (&glctx.tex_static, (char *)"pakutaso_sotsugyou.jpg");

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
#define MAX_BUF_COUNT 4

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

void
AppEngine::UpdateCameraTexture ()
{
    texture_2d_t *input_tex = &glctx.tex_camera;
    GLuint texid = input_tex->texid;
    int    cap_w, cap_h;
    void   *cap_buf;

    GetAppEngine()->AcquireCameraFrame (&cap_buf, &cap_w, &cap_h);

    if (cap_buf == NULL)
        return;

    if (texid == 0)
    {
        create_2d_texture_ex (input_tex, cap_buf, cap_w, cap_h, pixfmt_fourcc('R', 'G', 'B', 'A'));
    }
    else
    {
        glBindTexture (GL_TEXTURE_2D, texid);
        glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, cap_w, cap_h, GL_RGBA, GL_UNSIGNED_BYTE, cap_buf);
    }

    glctx.tex_camera_valid = true;
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

