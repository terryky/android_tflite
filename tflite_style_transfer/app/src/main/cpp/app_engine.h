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

#include "camera_manager.h"
#include "tflite_style_transfer.h"

typedef struct input_tex {
    GLuint texid;
    int    w, h;
    int    draw_x, draw_y, draw_w, draw_h;
} input_tex_t;

typedef struct gles_ctx {
    int initdone;
    int frame_count;

    char *str_glverstion;
    char *str_glvendor;
    char *str_glrender;

    int disp_w, disp_h;

    bool tex_camera_valid;
    input_tex_t tex_static;
    input_tex_t tex_camera;
    input_tex_t tex_style;
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
    
    void LoadInputTexture (input_tex_t *tex, char *fname);
    void UpdateCameraTexture ();
    void AdjustTexture (int win_w, int win_h, int texw, int texh,
                        int *dx, int *dy, int *dw, int *dh);

    void UpdateFrame (void);
    void RenderFrame (void);

    void DrawTFLiteConfigInfo ();

    // Style Transfer Specific
    void FeedInputImage (int is_predict, int texid, int win_w, int win_h);
    void FeedBlendStyle (style_predict_t *style0, style_predict_t *style1, float ratio);
    void StoreStylePredictParam (style_predict_t *style);


private:

    struct android_app  *m_app;

    bool                m_cameraGranted;
    NDKCamera           *m_camera;
    AImageReader        *m_img_reader;

    gles_ctx_t          glctx;
    std::vector<uint8_t> m_style_predict_tflite_model_buf;
    std::vector<uint8_t> m_style_transfer_tflite_model_buf;
    style_predict_t     m_style_predict[2];
    float               m_style_ratio;

public:
};

AppEngine *GetAppEngine (void);

#endif  // __APP_ENGINE_H__
