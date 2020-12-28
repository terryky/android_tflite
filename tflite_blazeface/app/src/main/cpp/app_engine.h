/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef __APP_ENGINE_H__
#define __APP_ENGINE_H__

#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <functional>
#include <thread>
#include <GLES2/gl2.h>

#include "util_texture.h"
#include "camera_manager.h"
#include "render_imgui.h"
#include "tflite_blazeface.h"

typedef struct gles_ctx {
    int initdone;
    int frame_count;

    char *str_glverstion;
    char *str_glvendor;
    char *str_glrender;

    int disp_w, disp_h;

    bool tex_camera_valid;
    texture_2d_t tex_static;
    texture_2d_t tex_camera;
} gles_ctx_t;


class AppEngine {
public:
    explicit AppEngine(android_app* app);
    ~AppEngine();

    // Interfaces to android application framework
    struct android_app* AndroidApp(void) const;
    void OnAppInitWindow(void);
    void OnAppTermWindow(void);

    // Manage Camera Permission
    void RequestCameraPermission();
    void OnCameraPermission(jboolean granted);

    // Manage NDKCamera Object
    void InitCamera (void);
    void CreateCamera(void);
    void DeleteCamera(void);
    void AcquireCameraFrame (void **cap_buf, int *cap_w, int *cap_h);

    // OpenGLES Render
    void InitGLES (void);
    void TerminateGLES (void);
    
    void LoadInputTexture (texture_2d_t *tex, char *fname);
    void UpdateCameraTexture ();
    void adjust_texture (int win_w, int win_h, int texw, int texh,
                         int *dx, int *dy, int *dw, int *dh);

    void UpdateFrame (void);
    void RenderFrame (void);

    void DrawTFLiteConfigInfo ();
    void setup_imgui (int win_w, int win_h, imgui_data_t *imgui_data);

    // Blazeface Specific
    void feed_blazeface_image (texture_2d_t *srctex, int win_w, int win_h);
    void render_detect_region (int ofstx, int ofsty, int texw, int texh,
                               blazeface_result_t *detection, imgui_data_t *imgui_data);

private:

    struct android_app  *m_app;

    bool                m_cameraGranted;
    NDKCamera           *m_camera;
    AImageReader        *m_img_reader;

    gles_ctx_t          glctx;
    std::vector<uint8_t> m_tflite_model_buf;

    imgui_data_t        imgui_data;

public:
};

AppEngine *GetAppEngine (void);

#endif  // __APP_ENGINE_H__
