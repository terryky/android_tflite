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
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES/gl.h>
#include <GLES/glext.h>
#include <GLES2/gl2.h>

#include "util_texture.h"
#include "util_render_target.h"
#include "camera_manager.h"
#include "render_imgui.h"
#include "tflite_blazeface.h"
#include "gestureDetector.h"

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
    texture_2d_t tex_input;
    EGLImage egl_img;

    render_target_t rtarget_main;
    render_target_t rtarget_crop;
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
    void CreateCamera (int facing);
    void DeleteCamera (void);

    // OpenGLES Render
    void InitGLES (void);
    void TerminateGLES (void);
    
    void LoadInputTexture (texture_2d_t *tex, char *fname);
    void UpdateCameraTexture ();
    void CropCameraTexture ();

    void UpdateFrame (void);
    void RenderFrame (void);

    void DrawTFLiteConfigInfo ();

    // IMGUI
    void setup_imgui (int win_w, int win_h, imgui_data_t *imgui_data);

    /* for touch gesture */
    ndk_helper::TapDetector        tap_detector_;
    ndk_helper::DoubletapDetector  doubletap_detector_;
    ndk_helper::DragDetector       drag_detector_;

    void mousemove_cb (int x, int y);
    void button_cb (int button, int state, int x, int y);
    void keyboard_cb (int key, int state, int x, int y);

private:

    struct android_app  *m_app;

    bool                m_cameraGranted;
    NDKCamera           *m_camera;
    ImageReaderHelper   m_ImgReader;

    gles_ctx_t          glctx;
    std::vector<uint8_t> m_tflite_model_buf;

    imgui_data_t        imgui_data;
    int                 m_camera_facing;

public:
};

AppEngine *GetAppEngine (void);

#endif  // __APP_ENGINE_H__
