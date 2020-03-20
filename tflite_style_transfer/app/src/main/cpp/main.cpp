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
#include "tflite_style_transfer.h"

#define UNUSED(x) (void)(x)

static std::vector<uint8_t> s_style_predict_tflite_model_buf;
static std::vector<uint8_t> s_style_transfer_tflite_model_buf;
static int s_tflite_stat = 0;


struct engine {
    struct android_app* app;

    int initdone;
    int frame_count;

    char *str_glverstion;
    char *str_glvendor;
    char *str_glrender;

    int disp_w, disp_h;

    GLuint style_texid;
    int style_texw, style_texh;

    GLuint texid;
    int texw, texh;
    int draw_x, draw_y, draw_w, draw_h;

    float style_ratio;
    style_predict_t style_predict[2];
};



void
load_asset_texture (struct engine* engine)
{
    int32_t img_w, img_h;
    uint8_t *img_buf;

    img_buf = asset_read_image (engine->app->activity->assetManager, (char *)"pakutaso_famicom.jpg", &img_w, &img_h);
    engine->texid = create_2d_texture ((void *)img_buf, img_w, img_h);
    asset_free_image (img_buf);

    engine->texw = img_w;
    engine->texh = img_h;

    /* load texture for style prediction */
    img_buf = asset_read_image (engine->app->activity->assetManager, (char *)"visual-cloud-03.jpg", &img_w, &img_h);
    engine->style_texid = create_2d_texture ((void *)img_buf, img_w, img_h);
    asset_free_image (img_buf);

    engine->style_texw = img_w;
    engine->style_texh = img_h;
}




/* resize image to DNN network input size and convert to fp32. */
void
feed_style_transfer_image(int is_predict, int texid, int win_w, int win_h)
{
#if defined (USE_INPUT_CAMERA_CAPTURE)
    update_capture_texture (texid);
#endif

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
store_style_predict (style_predict_t *style)
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

void
feed_blend_style (style_predict_t *style0, style_predict_t *style1, float ratio)
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
draw_tfilte_config_info (struct engine *engine)
{
    char strbuf[512];
    float col_pink[]  = {1.0f, 0.0f, 1.0f, 0.5f};
    float col_gray[]  = {0.5f, 0.5f, 0.5f, 0.5f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float *col_bg;
    
#if defined (USE_GPU_DELEGATEV2)
    sprintf (strbuf, "GPU_DELEGATEV2: ON ");
    col_bg = col_pink;
#else
    sprintf (strbuf, "GPU_DELEGATEV2: OFF");
    col_bg = col_gray;
#endif
    draw_dbgstr_ex (strbuf, engine->disp_w - 500, 0, 2.0f, col_white, col_bg);

#if defined (USE_QUANT_TFLITE_MODEL)
    sprintf (strbuf, "MODEL_INTQUANT: ON ");
    col_bg = col_pink;
    draw_dbgstr_ex (strbuf, engine->disp_w - 500, 48, 2.0f, col_white, col_bg);
#endif
}

static void 
engine_draw_frame(struct engine* engine) 
{
    int win_w = engine->disp_w;
    int win_h = engine->disp_h;
    int texid = engine->texid;
    int draw_x = engine->draw_x;
    int draw_y = engine->draw_y;
    int draw_w = engine->draw_w;
    int draw_h = engine->draw_h;
    int style_texid = engine->style_texid;
    static double ttime[10] = {0}, interval, invoke_ms;

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    int count = engine->frame_count;

    /* --------------------------------------- *
     *  Style prediction
     * --------------------------------------- */
    if (count == 0)
    {
        glClear (GL_COLOR_BUFFER_BIT);

        /* predict style of original image */
        glClear (GL_COLOR_BUFFER_BIT);
        feed_style_transfer_image (1, engine->texid, win_w, win_h);
        invoke_style_predict (&engine->style_predict[0]);
        store_style_predict (&engine->style_predict[0]);

        /* predict style of target image */
        glClear (GL_COLOR_BUFFER_BIT);
        feed_style_transfer_image (1, engine->style_texid, win_w, win_h);
        invoke_style_predict (&engine->style_predict[1]);
        store_style_predict (&engine->style_predict[1]);

        engine->style_ratio = -0.1f;
    }

    /* --------------------------------------- *
     *  Style transfer
     * --------------------------------------- */
    {
        style_transfer_t style_transfered = {};
        float style_ratio = engine->style_ratio;

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
        engine->style_ratio = style_ratio;

        /* feed style parameter and original image */
        feed_blend_style (&engine->style_predict[0], &engine->style_predict[1], style_ratio);
        feed_style_transfer_image (0, texid, win_w, win_h);

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

        draw_tfilte_config_info (engine);

        /* renderer info */
        draw_dbgstr (engine->str_glverstion, 10, 0);
        draw_dbgstr (engine->str_glvendor,   10, 22);
        draw_dbgstr (engine->str_glrender,   10, 44);

        draw_pmeter (0, 100);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]\nstyle_ratio=%.1f", 
                                interval, invoke_ms, style_ratio);
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
                    (char *)STYLE_PREDICT_MODEL_PATH, s_style_predict_tflite_model_buf);

    asset_read_file (engine->app->activity->assetManager, 
                    (char *)STYLE_TRANSFER_MODEL_PATH, s_style_transfer_tflite_model_buf);

    s_tflite_stat = init_tflite_style_transfer (
        (const char *)s_style_predict_tflite_model_buf.data(), s_style_predict_tflite_model_buf.size(),
        (const char *)s_style_transfer_tflite_model_buf.data(), s_style_transfer_tflite_model_buf.size());

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
