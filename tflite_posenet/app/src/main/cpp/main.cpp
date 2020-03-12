/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <time.h>
#include <GLES2/gl2.h>
#include <android_native_app_glue.h>
#include "util_debug.h"
#include "util_asset.h"
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "tflite_posenet.h"

#define UNUSED(x) (void)(x)

static std::vector<uint8_t> s_tflite_model_buf;
static int s_tflite_stat = 0;


struct engine {
    struct android_app* app;

    int initdone;
    int frame_count;

    char *str_glverstion;
    char *str_glvendor;
    char *str_glrender;

    int disp_w, disp_h;

    GLuint texid[100];
    int texw, texh;
    int draw_x, draw_y, draw_w, draw_h;
};



void
load_asset_texture (struct engine* engine)
{
    int32_t img_w, img_h;
    uint8_t *img_buf;

    img_buf = asset_read_image (engine->app->activity->assetManager, (char *)"pakutaso_person.jpg", &img_w, &img_h);
    engine->texid[0] = create_2d_texture ((void *)img_buf, img_w, img_h);
    asset_free_image (img_buf);

    engine->texw = img_w;
    engine->texh = img_h;
}




/* resize image to DNN network input size and convert to fp32. */
void
feed_posenet_image(int texid, ssbo_t *ssbo, int win_w, int win_h)
{
#if defined (USE_INPUT_CAMERA_CAPTURE)
    update_capture_texture (texid);
#endif

#if defined (USE_INPUT_SSBO)
    resize_texture_to_ssbo (texid, ssbo);
#else
    int x, y, w, h;
#if defined (USE_QUANT_TFLITE_MODEL)
    unsigned char *buf_u8 = (unsigned char *)get_posenet_input_buf (&w, &h);
#else
    float *buf_fp32 = (float *)get_posenet_input_buf (&w, &h);
#endif
    unsigned char *buf_ui8, *pui8;

    pui8 = buf_ui8 = (unsigned char *)malloc(w * h * 4);

    draw_2d_texture (texid, 0, win_h - h, w, h, 1);

    glPixelStorei (GL_PACK_ALIGNMENT, 1);
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
#if defined (USE_QUANT_TFLITE_MODEL)
            *buf_u8 ++ = r;
            *buf_u8 ++ = g;
            *buf_u8 ++ = b;
#else
            *buf_fp32 ++ = (float)(r - mean) / std;
            *buf_fp32 ++ = (float)(g - mean) / std;
            *buf_fp32 ++ = (float)(b - mean) / std;
#endif
        }
    }

    free (pui8);
#endif
    return;
}


/* render a bone of skelton. */
void
render_bone (int ofstx, int ofsty, int drw_w, int drw_h, 
             posenet_result_t *pose_ret, int pid, 
             enum pose_key_id id0, enum pose_key_id id1,
             float *col)
{
    float x0 = pose_ret->pose[pid].key[id0].x * drw_w + ofstx;
    float y0 = pose_ret->pose[pid].key[id0].y * drw_h + ofsty;
    float x1 = pose_ret->pose[pid].key[id1].x * drw_w + ofstx;
    float y1 = pose_ret->pose[pid].key[id1].y * drw_h + ofsty;
    float s0 = pose_ret->pose[pid].key[id0].score;
    float s1 = pose_ret->pose[pid].key[id1].score;

    /* if the confidence score is low, draw more transparently. */
    col[3] = (s0 + s1) * 0.5f;
    draw_2d_line (x0, y0, x1, y1, col, 5.0f);

    col[3] = 1.0f;
}

void
render_posenet_result (int x, int y, int w, int h, posenet_result_t *pose_ret)
{
    float col_red[]    = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_orange[] = {1.0f, 0.6f, 0.0f, 1.0f};
    float col_cyan[]   = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_lime[]   = {0.0f, 1.0f, 0.3f, 1.0f};
    float col_pink[]   = {1.0f, 0.0f, 1.0f, 1.0f};
    float col_blue[]   = {0.0f, 0.5f, 1.0f, 1.0f};
    
    for (int i = 0; i < pose_ret->num; i ++)
    {
        /* draw skelton */

        /* body */
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kRightShoulder, col_cyan);
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kLeftHip,       col_cyan);
        render_bone (x, y, w, h, pose_ret, i, kRightShoulder, kRightHip,      col_cyan);
        render_bone (x, y, w, h, pose_ret, i, kLeftHip,       kRightHip,      col_cyan);

        /* legs */
        render_bone (x, y, w, h, pose_ret, i, kLeftHip,       kLeftKnee,      col_pink);
        render_bone (x, y, w, h, pose_ret, i, kLeftKnee,      kLeftAnkle,     col_pink);
        render_bone (x, y, w, h, pose_ret, i, kRightHip,      kRightKnee,     col_blue);
        render_bone (x, y, w, h, pose_ret, i, kRightKnee,     kRightAnkle,    col_blue);
        
        /* arms */
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kLeftElbow,     col_orange);
        render_bone (x, y, w, h, pose_ret, i, kLeftElbow,     kLeftWrist,     col_orange);
        render_bone (x, y, w, h, pose_ret, i, kRightShoulder, kRightElbow,    col_lime  );
        render_bone (x, y, w, h, pose_ret, i, kRightElbow,    kRightWrist,    col_lime  );

        /* draw key points */
        for (int j = 0; j < kPoseKeyNum; j ++)
        {
            float keyx = pose_ret->pose[i].key[j].x * w + x;
            float keyy = pose_ret->pose[i].key[j].y * h + y;
            int r = 9;
            draw_2d_fillrect (keyx - (r/2), keyy - (r/2), r, r, col_red);
        }

#if defined (USE_FIREBALL_PARTICLE)
        {
            float x0 = pose_ret->pose[i].key[kRightWrist].x * w + x;
            float y0 = pose_ret->pose[i].key[kRightWrist].y * h + y;
            float x1 = pose_ret->pose[i].key[kLeftWrist].x * w + x;
            float y1 = pose_ret->pose[i].key[kLeftWrist].y * h + y;
            render_posenet_particle (x0, y0, x1, y1);
        }
#endif
    }

#if defined (USE_FACE_MASK)
    render_facemask (x, y, w, h, pose_ret);
#endif
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


static void 
engine_draw_frame(struct engine* engine) 
{
    int win_w = engine->disp_w;
    int win_h = engine->disp_h;
    int texid = engine->texid[0];
    int draw_x = engine->draw_x;
    int draw_y = engine->draw_y;
    int draw_w = engine->draw_w;
    int draw_h = engine->draw_h;
    ssbo_t *ssbo = NULL;
    static double ttime[10] = {0}, interval, invoke_ms;

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    int count = engine->frame_count;
    {
        posenet_result_t pose_ret = {};
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);

        /* invoke pose estimation using TensorflowLite */
        feed_posenet_image (texid, ssbo, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_posenet (&pose_ret);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        glClear (GL_COLOR_BUFFER_BIT);

#if defined (USE_INPUT_SSBO) /* for Debug. */
        /* visualize the contents of SSBO for input tensor. */
        visualize_ssbo (ssbo);
#endif
        /* visualize the object detection results. */
        draw_2d_texture (texid,  draw_x, draw_y, draw_w, draw_h, 0);
        render_posenet_result (draw_x, draw_y, draw_w, draw_h, &pose_ret);

        sprintf (strbuf, "TFLITE_STAT : %d", s_tflite_stat);
        draw_dbgstr (strbuf, engine->disp_w - 300, 0);

        /* renderer info */
        draw_dbgstr (engine->str_glverstion, 10, 0);
        draw_dbgstr (engine->str_glvendor,   10, 22);
        draw_dbgstr (engine->str_glrender,   10, 44);

        draw_pmeter (0, 100);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]", interval, invoke_ms);
        draw_dbgstr (strbuf, 10, 80);

        egl_swap();
    }
}


static int
engine_init_display(struct engine* engine) 
{
    egl_init_with_window_surface (2, engine->app->window, 8, 0, 0);

    engine->str_glverstion = (char *)glGetString (GL_VERSION);
    engine->str_glvendor   = (char *)glGetString (GL_VENDOR);
    engine->str_glrender   = (char *)glGetString (GL_RENDERER);

    int w, h;
    egl_get_current_surface_dimension (&w, &h);

    init_2d_renderer (w, h);
    init_pmeter (w, h, h - 100);
    init_dbgstr (w, h);

    load_asset_texture (engine);
    adjust_texture (w, h, engine->texw, engine->texh, 
                    &engine->draw_x, &engine->draw_y, &engine->draw_w, &engine->draw_h);

    engine->disp_w = w;
    engine->disp_h = h;

    asset_read_file (engine->app->activity->assetManager, 
                    (char *)POSENET_MODEL_PATH, s_tflite_model_buf);

    s_tflite_stat = init_tflite_posenet (NULL, (const char *)s_tflite_model_buf.data(), 
                                         s_tflite_model_buf.size());

    engine->initdone = 1;

    return 0;
}

static void
engine_term_display(struct engine* engine)
{
    egl_terminate ();
}

static void
engine_handle_cmd(struct android_app* app, int32_t cmd) 
{
    struct engine* engine = (struct engine*)app->userData;
    switch (cmd) {
    // The window is being shown, get it ready.
    case APP_CMD_INIT_WINDOW: 
        if (engine->app->window != NULL) 
        {
            engine_init_display (engine);
        }
        break;
    // The window is being hidden or closed, clean it up.
    case APP_CMD_TERM_WINDOW: 
        engine_term_display (engine);
        break;
    }
}


/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
void android_main(struct android_app* state) 
{
    struct engine engine = {0};

    state->userData = &engine;
    state->onAppCmd = engine_handle_cmd;
    engine.app = state;

    while (1)
    {
        int ident, events;
        struct android_poll_source* source;

        while ((ident = ALooper_pollAll(0, NULL, &events, (void**)&source)) >= 0)
        {
            if (source != NULL) {
                source->process(state, source);
            }

            // Check if we are exiting.
            if (state->destroyRequested != 0) {
                engine_term_display(&engine);
                return;
            }
        }

        if (engine.initdone) {
            engine_draw_frame(&engine);
            engine.frame_count++;
        }
    }
}
