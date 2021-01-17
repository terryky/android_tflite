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
#include "render_dense_depth.h"
#include "touch_event.h"
#include "render_imgui.h"
#include "assertgl.h"

#define UNUSED(x) (void)(x)

#define CAMERA_RESOLUTION_W     640
#define CAMERA_RESOLUTION_H     480
#define CAMERA_CROP_WIDTH       480 /* make a src image square */
#define CAMERA_CROP_HEIGHT      480 /* make a src image square */

static imgui_data_t s_gui_prop = {0};




/* resize image to DNN network input size and convert to fp32. */
void
feed_dense_depth_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_dense_depth_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    draw_2d_texture_ex (srctex, 0, win_h - h, w, h, RENDER2D_FLIP_V);

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



float
fclampf (float val)
{
    val = fmaxf (0.0f, val);
    val = fminf (1.0f, val);
    return val;
}

static void
render_depth_image (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh,
                    dense_depth_result_t *dense_depth_ret)
{
    float *depthmap = dense_depth_ret->depthmap;
    int depthmap_w  = dense_depth_ret->depthmap_dims[0];
    int depthmap_h  = dense_depth_ret->depthmap_dims[1];
    int x, y;
    unsigned int imgbuf[depthmap_h][depthmap_w];

    /* find the most confident class for each pixel. */
    for (y = 0; y < depthmap_h; y ++)
    {
        for (x = 0; x < depthmap_w; x ++)
        {
            float d = depthmap[y * depthmap_w + x];
            d -= 0.0;
            d /= 10.0;
            d = fclampf (d);

            unsigned char r = d * 255;
            unsigned char g = r;
            unsigned char b = r;
            unsigned char a = 255;

            imgbuf[y][x] = (a << 24) | (b << 16) | (g << 8) | (r);
        }
    }

    texture_2d_t animtex;
    create_2d_texture_ex (&animtex, imgbuf, depthmap_w, depthmap_h, pixfmt_fourcc ('R', 'G', 'B', 'A'));
    draw_2d_texture_ex (&animtex, ofstx, ofsty, texw, texh, 0);

    glDeleteTextures (1, &animtex.texid);
}


static void
render_depth_image_3d (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh,
                       dense_depth_result_t *dense_depth_ret)
{
    float mtxGlobal[16], mtxTouch[16];
    static int s_is_first_render3d = 1;
    static mesh_obj_t s_depth_mesh;

    get_touch_event_matrix (mtxTouch);
    matrix_identity (mtxGlobal);
    matrix_translate (mtxGlobal, 0, 0, -s_gui_prop.camera_pos_z);
    matrix_mult (mtxGlobal, mtxGlobal, mtxTouch);

    float *depthmap = dense_depth_ret->depthmap;
    int depthmap_w  = dense_depth_ret->depthmap_dims[0];
    int depthmap_h  = dense_depth_ret->depthmap_dims[1];

    /* create mesh object */
    if (s_is_first_render3d)
    {
        create_mesh (&s_depth_mesh, depthmap_w - 1, depthmap_h - 1);
        s_is_first_render3d = 0;
    }
    float *vtx = s_depth_mesh.vtx_array;
    float *uv  = s_depth_mesh.uv_array;

    /* create 3D vertex coordinate */
    for (int y = 0; y < depthmap_h; y ++)
    {
        for (int x = 0; x < depthmap_w; x ++)
        {
            int   idx = (y * depthmap_w + x);
            float d = depthmap[idx];

            if (1)
            {
                d -= 0;//s_gui_prop.depth_min;
                d /= 10;//s_gui_prop.depth_max;
                d = (d * 2.0 - 1.0) * s_gui_prop.pose_scale_z;
            }
            else
            {
                //d = s_gui_prop.depth_max / d;   //  inf -> 1.0
                //d = 2 - d;                      // -inf -> 1.0
                //d = d * s_gui_prop.pose_scale_z;
            }

            vtx[3 * idx + 0] =  ((x / (float)depthmap_h) * 2.0f - 1.0f) * s_gui_prop.pose_scale_x;
            vtx[3 * idx + 1] = -((y / (float)depthmap_h) * 2.0f - 1.0f) * s_gui_prop.pose_scale_y;
            vtx[3 * idx + 2] =  d;

            uv [2 * idx + 0] = x / (float)depthmap_w;
            uv [2 * idx + 1] = y / (float)depthmap_h;
        }
    }
    float colb[] = {1.0, 1.0, 1.0, 1.0};
    draw_point_arrays (mtxGlobal, vtx, uv, depthmap_h * depthmap_w, srctex->texid, colb);

    if (s_gui_prop.draw_axis)
    {
        /* (xyz)-AXIS */
        for (int i = -1; i <= 1; i ++)
        {
            for (int j = -1; j <= 1; j ++)
            {
                float col_base[] = {0.1, 0.5, 0.5, 0.5};
                float dx = s_gui_prop.pose_scale_x;
                float dy = s_gui_prop.pose_scale_y;
                float dz = s_gui_prop.pose_scale_z;

                {
                    float v0[3] = {-dx, i * dy, j * dz};
                    float v1[3] = { dx, i * dy, j * dz};
                    float col_red[] = {1.0, 0.0, 0.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_red : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                }
                {
                    float v0[3] = {i * dx, -dy, j * dz};
                    float v1[3] = {i * dx,  dy, j * dz};
                    float col_green[] = {0.0, 1.0, 0.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_green : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                }
                {
                    float v0[3] = {i * dx, j * dy, -dz};
                    float v1[3] = {i * dx, j * dy,  dz};
                    float col_blue[] = {0.0, 0.0, 1.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_blue : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                }
            }
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

    touch_event_move (0, x, y);
}

void
AppEngine::button_cb (int button, int state, int x, int y)
{
    imgui_mousebutton (button, state, x, y);

    if (state)
        touch_event_start (0, x, y);
    else
        touch_event_end (0);
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

    init_touch_event (win_w, win_h);

    init_imgui (win_w, win_h);
#endif

    imgui_data->camera_facing  = m_camera_facing;
    s_gui_prop.pose_scale_x = 100.0f;
    s_gui_prop.pose_scale_y = 100.0f;
    s_gui_prop.pose_scale_z = 100.0f;
    s_gui_prop.camera_pos_z = 200.0f;
    s_gui_prop.draw_axis    = 0;
    s_gui_prop.draw_pmeter  = 1;
    *imgui_data = s_gui_prop;
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

    s_gui_prop = imgui_data;
    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
    int count = glctx.frame_count;
    {
        dense_depth_result_t dense_depth_result = {0};
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);

        /* --------------------------------------- *
         *  Dense Depth
         * --------------------------------------- */
        feed_dense_depth_image (&srctex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_dense_depth (&dense_depth_result);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  render scene
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render_depth_image_3d (&srctex, draw_x, draw_y, draw_w, draw_h, &dense_depth_result);

        float camx = 100;
        float camy = 100;
        float camw = 300;
        float camh = 300;
        /* visualize the hand pose estimation results. */
        draw_2d_texture_ex (&srctex, camx, camy, camw, camh, 0);

        int dx = camx;
        int dy = camy + camh;
        int dw = camw;
        int dh = camh;
        render_depth_image (&srctex, dx, dy, dw, dh, &dense_depth_result);

        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        glViewport (0, 0, win_w, win_h);

        DrawTFLiteConfigInfo ();

        if (s_gui_prop.draw_pmeter)
        {
            draw_pmeter (0, 40);
        }

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
    init_pmeter (w, h, h - 100);
    init_dbgstr (w, h);

    init_cube ((float)w / (float)h);
    asset_read_file (m_app->activity->assetManager,
                    (char *)DENSEDEPTH_MODEL_PATH, m_tflite_model_buf);

    ret = init_tflite_dense_depth (
        (const char *)m_tflite_model_buf.data(), m_tflite_model_buf.size());

    setup_imgui (w, h, &imgui_data);

    glctx.disp_w = w;
    glctx.disp_h = h;
    LoadInputTexture (&glctx.tex_static, (char *)"pexels.jpg");

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

