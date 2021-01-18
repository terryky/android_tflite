/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <cstdio>
#include "util_debug.h"
#include "util_asset.h"
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "util_matrix.h"
#include "app_engine.h"
#include "render_hair.h"
#include "render_imgui.h"
#include "assertgl.h"

#define UNUSED(x) (void)(x)

#define CAMERA_RESOLUTION_W     640
#define CAMERA_RESOLUTION_H     480
#define CAMERA_CROP_WIDTH       480 /* make a src image square */
#define CAMERA_CROP_HEIGHT      480 /* make a src image square */


/* resize image to DNN network input size and convert to fp32. */
void
feed_segmentation_image (texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_segmentation_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    draw_2d_texture_ex (srctex, 0, win_h - h, w, h, RENDER2D_FLIP_V);

    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

    /* convert UI8 [0, 255] ==> FP32 [0, 1] */
    float mean =   0.0f;
    float std  = 255.0f;
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
            *buf_fp32 ++ = 0.0f;
        }
    }

    return;
}

static void 
colormap_hsv (float h, float col[4])
{
    float s = 1.0f;
    float v = 1.0f;

    float r = v;
    float g = v;
    float b = v;

    h *= 6.0f;
    int i = (int) h;
    float f = h - (float) i;

    switch (i) 
    {
    default:
    case 0:
        g *= 1 - s * (1 - f);
        b *= 1 - s;
        break;
    case 1:
        r *= 1 - s * f;
        b *= 1 - s;
        break;
    case 2:
        r *= 1 - s;
        b *= 1 - s * (1 - f);
        break;
    case 3:
        r *= 1 - s;
        g *= 1 - s * f;
        break;
    case 4:
        r *= 1 - s * (1 - f);
        g *= 1 - s;
        break;
    case 5:
        g *= 1 - s;
        b *= 1 - s * f;
        break;
    }

    col[0] = r;
    col[1] = g;
    col[2] = b;
    col[3] = 1.0f;
}

static void
render_hsv_circle (int x0, int y0, float angle)
{
    float col[4];
    float r = 80.0f;

    float gray[] = {0.0f, 0.0f, 0.0f, 0.2f};
    draw_2d_fillrect (x0 - r - 10, y0 - r - 10, 2 * r + 20, 2 * r + 20, gray);

    for (int i = 0; i < 360; i ++)
    {
        int x1 = x0 + (int)(r * cosf (DEG_TO_RAD(i)) + 0.5f);
        int y1 = y0 + (int)(r * sinf (DEG_TO_RAD(i)) + 0.5f);

        colormap_hsv ((float)i /360.0f, col);
        draw_2d_line (x0, y0, x1, y1, col, 2);
    }

    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int x1 = x0 + (int)(r * cosf (DEG_TO_RAD(angle)) + 0.5f);
    int y1 = y0 + (int)(r * sinf (DEG_TO_RAD(angle)) + 0.5f);
    draw_2d_line (x0, y0, x1, y1, col_white, 2);

    int q = 10;
    draw_2d_rect (x1 - (q/2), y1 - (q/2), q, q, col_white, 2);
}

//#define RENDER_BY_BLEND 1
void
render_segment_result (int ofstx, int ofsty, int draw_w, int draw_h, 
                       texture_2d_t *srctex, segmentation_result_t *segment_ret)
{
    float *segmap = segment_ret->segmentmap;
    int segmap_w  = segment_ret->segmentmap_dims[0];
    int segmap_h  = segment_ret->segmentmap_dims[1];
    int segmap_c  = segment_ret->segmentmap_dims[2];
    int x, y, c;
    static unsigned int *imgbuf = NULL;
    float hair_color[4] = {0};
    float back_color[4] = {0};
    static float s_hsv_h = 0.0f;

    if (imgbuf == NULL)
    {
        imgbuf = (unsigned int *)malloc (segmap_w * segmap_h * sizeof(unsigned int));
    }

    s_hsv_h += 5.0f;
    if (s_hsv_h >= 360.0f)
        s_hsv_h = 0.0f;

    colormap_hsv (s_hsv_h / 360.0f, hair_color);

#if defined (RENDER_BY_BLEND)
    float lumi = (hair_color[0] * 0.299f + hair_color[1] * 0.587f + hair_color[2] * 0.114f);
    hair_color[3] = lumi;
#endif

    /* find the most confident class for each pixel. */
    for (y = 0; y < segmap_h; y ++)
    {
        for (x = 0; x < segmap_w; x ++)
        {
            int max_id;
            float conf_max = 0;
            for (c = 0; c < MAX_SEGMENT_CLASS; c ++)
            {
                float confidence = segmap[(y * segmap_w * segmap_c)+ (x * segmap_c) + c];
                if (c == 0 || confidence > conf_max)
                {
                    conf_max = confidence;
                    max_id = c;
                }
            }

            float *col = (max_id > 0) ? hair_color : back_color;
            unsigned char r = ((int)(col[0] * 255)) & 0xff;
            unsigned char g = ((int)(col[1] * 255)) & 0xff;
            unsigned char b = ((int)(col[2] * 255)) & 0xff;
            unsigned char a = ((int)(col[3] * 255)) & 0xff;

            imgbuf[y * segmap_w + x] = (a << 24) | (b << 16) | (g << 8) | (r);
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

#if !defined (RENDER_BY_BLEND)
    draw_colored_hair (srctex, texid, ofstx, ofsty, draw_w, draw_h, 0, hair_color);
#else
    draw_2d_texture_ex (srctex, ofstx, ofsty, draw_w, draw_h, 0);

    unsigned int blend_add  [] = {GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE};
    draw_2d_texture_blendfunc (texid, ofstx, ofsty, draw_w, draw_h, 0, blend_add);
#endif

    glDeleteTextures (1, &texid);

    render_hsv_circle (ofstx + draw_w - 100, ofsty + 100, s_hsv_h);
}



void
AppEngine::DrawTFLiteConfigInfo ()
{
    char strbuf[512];
    float col_pink[]  = {1.0f, 0.0f, 1.0f, 0.5f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float *col_bg = col_pink;

#if defined (USE_GPU_DELEGATEV2)
    sprintf (strbuf, "GPU_DELEGATEV2: ON ");
#else
    sprintf (strbuf, "GPU_DELEGATEV2: ---");
#endif
    draw_dbgstr_ex (strbuf, glctx.disp_w - 250, glctx.disp_h - 24, 1.0f, col_white, col_bg);

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
static void
adjust_texture (int win_w, int win_h, int texw, int texh,
                int *dx, int *dy, int *dw, int *dh, int full_zoom)
{
    float win_aspect = (float)win_w / (float)win_h;
    float tex_aspect = (float)texw  / (float)texh;
    float scale;
    float scaled_w, scaled_h;
    float offset_x, offset_y;

    if (((full_zoom == 0) && (win_aspect > tex_aspect)) ||
        ((full_zoom == 1) && (win_aspect < tex_aspect)) )
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

    imgui_data->camera_facing  = m_camera_facing;
}


void 
AppEngine::RenderFrame ()
{
    texture_2d_t srctex = glctx.tex_input;
    int win_w  = glctx.disp_w;
    int win_h  = glctx.disp_h;
    static double ttime[10] = {0}, interval, invoke_ms;

    int draw_x, draw_y, draw_w, draw_h;
    int texw = srctex.width;
    int texh = srctex.height;
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h, 0);

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
    int count = glctx.frame_count;
    {
        segmentation_result_t segment_result;
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);

        /* --------------------------------------- *
         *  hair segmentation
         * --------------------------------------- */
        feed_segmentation_image (&srctex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_segmentation (&segment_result);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  render scene
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT);
        draw_2d_texture_ex (&srctex, draw_x, draw_y, draw_w, draw_h, 0);

        render_segment_result (draw_x, draw_y, draw_w, draw_h, &srctex, &segment_result);

        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        DrawTFLiteConfigInfo ();

        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]", interval, invoke_ms);
        draw_dbgstr (strbuf, 10, 10);

        /* renderer info */
        int y = win_h - 22 * 3;
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
      m_camera_facing(0)
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
    init_hair_renderer (w, h);
    init_pmeter (w, h, h - 100);
    init_dbgstr (w, h);

    asset_read_file (m_app->activity->assetManager,
                    (char *)SEGMENTATION_MODEL_PATH, m_tflite_model_buf);

    ret = init_tflite_segmentation (
        (const char *)m_tflite_model_buf.data(), m_tflite_model_buf.size());

    setup_imgui (w, h, &imgui_data);

    glctx.disp_w = w;
    glctx.disp_h = h;
    LoadInputTexture (&glctx.tex_static, (char *)"ride_horse.jpg");

    /* render target for default framebuffer */
    get_render_target (&glctx.rtarget_main);

    /* render target for camera cropping */
    create_render_target (&glctx.rtarget_crop, CAMERA_CROP_WIDTH, CAMERA_CROP_HEIGHT, RTARGET_COLOR);
    glctx.tex_input.texid  = glctx.rtarget_crop.texc_id;
    glctx.tex_input.width  = glctx.rtarget_crop.width;
    glctx.tex_input.height = glctx.rtarget_crop.height;
    glctx.tex_input.format = pixfmt_fourcc('R', 'G', 'B', 'A');

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
        if (m_camera_facing != imgui_data.camera_facing)
        {
            m_camera_facing = imgui_data.camera_facing;
            DeleteCamera ();
            CreateCamera (m_camera_facing);
        }
        UpdateCameraTexture();
    }

    if (m_cameraGranted && glctx.tex_camera_valid == false)
        return;

    CropCameraTexture ();

    RenderFrame();
}

void
AppEngine::CropCameraTexture (void)
{
    texture_2d_t srctex = glctx.tex_camera;
    if (!glctx.tex_camera_valid)
        srctex = glctx.tex_static;

    /* render to square FBO */
    render_target_t *rtarget = &glctx.rtarget_crop;
    set_render_target (rtarget);
    set_2d_projection_matrix (rtarget->width, rtarget->height);
    glClear (GL_COLOR_BUFFER_BIT);

    int draw_x, draw_y, draw_w, draw_h;
    adjust_texture (rtarget->width, rtarget->height, srctex.width, srctex.height,
                    &draw_x, &draw_y, &draw_w, &draw_h, 1);

    /* when we use inner camera, enable horizontal flip. */
    int flip = m_camera_facing ? RENDER2D_FLIP_H : 0;
    flip |= RENDER2D_FLIP_V;
    draw_2d_texture_ex (&srctex, draw_x, draw_y, draw_w, draw_h, flip);

    /* reset to the default framebuffer */
    rtarget = &glctx.rtarget_main;
    set_render_target (rtarget);
    set_2d_projection_matrix (rtarget->width, rtarget->height);
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

    CreateCamera (m_camera_facing);
}


void
AppEngine::DeleteCamera(void)
{
    if (m_camera)
    {
        delete m_camera;
        m_camera = nullptr;
    }

    m_ImgReader.ReleaseImageReader ();
    glctx.tex_camera_valid = false;
}


void 
AppEngine::CreateCamera (int facing)
{
    m_camera = new NDKCamera();
    ASSERT (m_camera, "Failed to Create CameraObject");

    m_camera->SelectCameraFacing (facing);

    m_ImgReader.InitImageReader (CAMERA_RESOLUTION_W, CAMERA_RESOLUTION_H);
    ANativeWindow *nativeWindow = m_ImgReader.GetNativeWindow();

    m_camera->CreateSession (nativeWindow);
    m_camera->StartPreview (true);
}

void
AppEngine::UpdateCameraTexture ()
{
    /* Acquire the latest AHardwareBuffer */
    AHardwareBuffer *ahw_buf = NULL;
    int ret = m_ImgReader.GetCurrentHWBuffer (&ahw_buf);
    if (ret != 0)
        return;

    /* Get EGLClientBuffer */
    EGLClientBuffer egl_buf = eglGetNativeClientBufferANDROID (ahw_buf);
    if (!egl_buf)
    {
        DBG_LOGE("Failed to create EGLClientBuffer");
        return;
    }

    /* (Re)Create EGLImage */
    if (glctx.egl_img != EGL_NO_IMAGE_KHR)
    {
        eglDestroyImageKHR (egl_get_display(), glctx.egl_img);
        glctx.egl_img = EGL_NO_IMAGE_KHR;
    }

    EGLint attrs[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE,};
    glctx.egl_img = eglCreateImageKHR (egl_get_display(), EGL_NO_CONTEXT,
                                       EGL_NATIVE_BUFFER_ANDROID, egl_buf, attrs);

    /* Bind to GL_TEXTURE_EXTERNAL_OES */
    texture_2d_t *input_tex = &glctx.tex_camera;
    if (input_tex->texid == 0)
    {
        GLuint texid;
        glGenTextures (1, &texid);
        glBindTexture (GL_TEXTURE_EXTERNAL_OES, texid);

        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        input_tex->texid  = texid;
        input_tex->format = pixfmt_fourcc('E', 'X', 'T', 'X');
        m_ImgReader.GetBufferDimension (&input_tex->width, &input_tex->height);
    }
    glBindTexture (GL_TEXTURE_EXTERNAL_OES, input_tex->texid);

    glEGLImageTargetTexture2DOES (GL_TEXTURE_EXTERNAL_OES, glctx.egl_img);
    GLASSERT ();

    glctx.tex_camera_valid = true;
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

