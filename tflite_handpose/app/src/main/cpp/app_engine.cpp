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
#include "render_handpose.h"
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
feed_palm_detection_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_palm_detection_input_buf (&w, &h);
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

void
feed_hand_landmark_image(texture_2d_t *srctex, int win_w, int win_h, palm_detection_result_t *detection, unsigned int hand_id)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_hand_landmark_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    float texcoord[] = { 0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 0.0f };

    if (detection->num > hand_id)
    {
        palm_t *palm = &(detection->palms[hand_id]);
        float x0 = palm->hand_pos[0].x;
        float y0 = palm->hand_pos[0].y;
        float x1 = palm->hand_pos[1].x; //    0--------1
        float y1 = palm->hand_pos[1].y; //    |        |
        float x2 = palm->hand_pos[2].x; //    |        |
        float y2 = palm->hand_pos[2].y; //    3--------2
        float x3 = palm->hand_pos[3].x;
        float y3 = palm->hand_pos[3].y;
        texcoord[0] = x3;   texcoord[1] = y3;
        texcoord[2] = x0;   texcoord[3] = y0;
        texcoord[4] = x2;   texcoord[5] = y2;
        texcoord[6] = x1;   texcoord[7] = y1;
    }

    draw_2d_texture_ex_texcoord (srctex, 0, win_h - h, w, h, texcoord);

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


static void
render_palm_region (int ofstx, int ofsty, int texw, int texh, palm_t *palm)
{
    float col_blue[]  = {0.0f, 0.0f, 1.0f, 1.0f};
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    float x1 = palm->rect.topleft.x  * texw + ofstx;
    float y1 = palm->rect.topleft.y  * texh + ofsty;
    float x2 = palm->rect.btmright.x * texw + ofstx;
    float y2 = palm->rect.btmright.y * texh + ofsty;
    float score = palm->score;

    /* detect rectangle */
    draw_2d_rect (x1, y1, x2-x1, y2-y1, col_blue, 2.0f);

    /* detect score */
    char buf[512];
    sprintf (buf, "%d", (int)(score * 100));
    draw_dbgstr_ex (buf, x1, y1, 1.0f, col_white, col_blue);

    /* key points */
    for (int j = 0; j < 7; j ++)
    {
        float x = palm->keys[j].x * texw + ofstx;
        float y = palm->keys[j].y * texh + ofsty;

        int r = 4;
        draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_blue);
    }

    /* ROI rectangle */
    for (int j0 = 0; j0 < 4; j0 ++)
    {
        int j1 = (j0 + 1) % 4;
        float x1 = palm->hand_pos[j0].x * texw + ofstx;
        float y1 = palm->hand_pos[j0].y * texh + ofsty;
        float x2 = palm->hand_pos[j1].x * texw + ofstx;
        float y2 = palm->hand_pos[j1].y * texh + ofsty;

        draw_2d_line (x1, y1, x2, y2, col_red, 2.0f);
    }
}

static void
render_2d_bone (int ofstx, int ofsty, int texw, int texh, hand_landmark_result_t *hand_landmark,
                int id0, int id1)
{
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float x0 = hand_landmark->joint[id0].x * texw + ofstx;
    float y0 = hand_landmark->joint[id0].y * texh + ofsty;
    float x1 = hand_landmark->joint[id1].x * texw + ofstx;
    float y1 = hand_landmark->joint[id1].y * texh + ofsty;

    draw_2d_line (x0, y0, x1, y1, col_white, 1.0f);
}

static void
compute_2d_skelton_pos (hand_landmark_result_t *dst_hand, hand_landmark_result_t *src_hand, palm_t *palm)
{
    float rotation = RAD_TO_DEG (palm->rotation);  /* z rotation (from detection result) */
    float ofset_x = palm->hand_cx;
    float ofset_y = palm->hand_cy;
    float scale_w = palm->hand_w;
    float scale_h = palm->hand_h;

    float mtx[16];
    matrix_identity (mtx);
    matrix_translate (mtx, ofset_x, ofset_y, 0.0f);
    matrix_rotate (mtx, rotation, 0.0f, 0.0f, 1.0f);
    matrix_scale (mtx, scale_w, scale_h, 1.0f);
    matrix_translate (mtx, -0.5f, -0.5f, 0.0f);

    /* multiply rotate matrix */
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        float x = src_hand->joint[i].x;
        float y = src_hand->joint[i].y;

        float vec[2] = {x, y};
        matrix_multvec2 (mtx, vec, vec);

        dst_hand->joint[i].x = vec[0];
        dst_hand->joint[i].y = vec[1];
    }
}

static void
render_skelton_2d (int ofstx, int ofsty, int texw, int texh, palm_t *palm,
                   hand_landmark_result_t *hand_landmark)
{
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_cyan[]  = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    /* transform to global coordinate */
    hand_landmark_result_t hand_draw;
    compute_2d_skelton_pos (&hand_draw, hand_landmark, palm);

    /* score of keypoint */
    {
        float x = hand_draw.joint[0].x * texw + ofstx;
        float y = hand_draw.joint[0].y * texh + ofsty;
        float score = hand_landmark->score;
        char buf[512];
        sprintf (buf, "key:%d", (int)(score * 100));
        draw_dbgstr_ex (buf, x, y, 1.0f, col_white, col_red);
    }

    /* keypoints */
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        float x = hand_draw.joint[i].x  * texw + ofstx;
        float y = hand_draw.joint[i].y  * texh + ofsty;

        int r = 4;
        draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_cyan);
    }

    /* skeltons */
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  0,  1);
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  0, 17);

    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  1,  5);
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  5,  9);
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  9, 13);
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw, 13, 17);

    for (int i = 0; i < 5; i ++)
    {
        int idx0 = 4 * i + 1;
        int idx1 = idx0 + 1;
        render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw, idx0,   idx1);
        render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw, idx0+1, idx1+1);
        render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw, idx0+2, idx1+2);
    }
}


static void
compute_3d_skelton_pos (hand_landmark_result_t *dst_hand, hand_landmark_result_t *src_hand, palm_t *palm)
{
    float xoffset = palm->hand_cx - 0.5f;
    float yoffset = palm->hand_cy - 0.5f;
    float zoffset = (1.0f - palm->hand_w) / (palm->hand_w) * 0.1f; /* (1/w - 1) = (1-w)/w */

    xoffset *= (1.0f + zoffset);
    yoffset *= (1.0f + zoffset);

    float rotation = -RAD_TO_DEG (palm->rotation);  /* z rotation (from detection result) */

    float mtx[16];
    matrix_identity (mtx);
    matrix_rotate (mtx, rotation, 0.0f, 0.0f, 1.0f);

    //fprintf (stderr, "hand_w = %f, zoffset = %f\n", palm->hand_w, zoffset);
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        float x = src_hand->joint[i].x;
        float y = src_hand->joint[i].y;
        float z = src_hand->joint[i].z;

        x = (x + xoffset) - 0.5f;
        y = (y + yoffset) - 0.5f;
        z = (z + zoffset);
        x = x * s_gui_prop.pose_scale_x * 2;
        y = y * s_gui_prop.pose_scale_y * 2;
        z = z * s_gui_prop.pose_scale_z * 5;
        y = -y;
        z = -z;

        /* multiply rotate matrix */
        {
            float vec[2] = {x, y};
            matrix_multvec2 (mtx, vec, vec);
            x = vec[0];
            y = vec[1];
        }

        dst_hand->joint[i].x = x;
        dst_hand->joint[i].y = y;
        dst_hand->joint[i].z = z;
    }
}

static void
render_3d_bone (float *mtxGlobal, hand_landmark_result_t *pose, int idx0, int idx1, 
                float *color, float rad, int is_shadow)
{
    float *pos0 = (float *)&(pose->joint[idx0]);
    float *pos1 = (float *)&(pose->joint[idx1]);

    draw_bone (mtxGlobal, pos0, pos1, rad, color, is_shadow);
}

static void
render_palm_tri (float *mtxGlobal, hand_landmark_result_t *hand_landmark, int idx0, int idx1, int idx2, float *color)
{
    float *pos0 = (float *)&hand_landmark->joint[idx0];
    float *pos1 = (float *)&hand_landmark->joint[idx1];
    float *pos2 = (float *)&hand_landmark->joint[idx2];

    draw_triangle (mtxGlobal, pos0, pos1, pos2, color);
}

static void
shadow_matrix (float *m, float *light_dir, float *ground_pos, float *ground_nrm)
{
    vec3_normalize (light_dir);
    vec3_normalize (ground_nrm);

    float a = ground_nrm[0];
    float b = ground_nrm[1];
    float c = ground_nrm[2];
    float d = 0;
    float ex = light_dir[0];
    float ey = light_dir[1];
    float ez = light_dir[2];

    m[ 0] =  b * ey + c * ez;
    m[ 1] = -a * ey;
    m[ 2] = -a * ez;
    m[ 3] = 0;

    m[ 4] = -b * ex;
    m[ 5] =  a * ex + c * ez;
    m[ 6] = -b * ez;
    m[ 7] = 0;

    m[ 8] = -c * ex;
    m[ 9] = -c * ey;
    m[10] =  a * ex + b * ey;
    m[11] = 0;

    m[12] = -d * ex;
    m[13] = -d * ey;
    m[14] = -d * ez;
    m[15] =  a * ex + b * ey + c * ey;
}

static void
render_skelton_3d (int ofstx, int ofsty, hand_landmark_result_t *hand_landmark, palm_t *palm)
{
    float mtxGlobal[16], mtxTouch[16];
    float col_red []   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_yellow[] = {1.0f, 1.0f, 0.0f, 1.0f};
    float col_green [] = {0.0f, 1.0f, 0.0f, 1.0f};
    float col_cyan  [] = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_violet[] = {1.0f, 0.0f, 1.0f, 1.0f};
    float col_palm[]   = {0.8f, 0.8f, 0.8f, 0.8f};
    float col_gray[]   = {0.0f, 0.0f, 0.0f, 0.5f};
    float col_node[]   = {1.0f, 1.0f, 1.0f, 1.0f};

    get_touch_event_matrix (mtxTouch);

    /* transform to 3D coordinate */
    hand_landmark_result_t hand_draw;
    compute_3d_skelton_pos (&hand_draw, hand_landmark, palm);


    for (int is_shadow = 1; is_shadow >= 0; is_shadow --)
    {
        float *colj;
        float *coln = col_node;
        float *colp = col_palm;

        matrix_identity (mtxGlobal);
        matrix_translate (mtxGlobal, 0.0, 0.0, -s_gui_prop.camera_pos_z);
        matrix_mult (mtxGlobal, mtxGlobal, mtxTouch);

        if (is_shadow)
        {
            float mtxShadow[16];
            float light_dir[3]  = {1.0f, 2.0f, 1.0f};
            float ground_pos[3] = {0.0f, 0.0f, 0.0f};
            float ground_nrm[3] = {0.0f, 1.0f, 0.0f};

            shadow_matrix (mtxShadow, light_dir, ground_pos, ground_nrm);

            float shadow_y = - s_gui_prop.pose_scale_y;
            //shadow_y += pose->key3d[kNeck].y * 0.5f;
            matrix_translate (mtxGlobal, 0.0, shadow_y, 0);
            matrix_mult (mtxGlobal, mtxGlobal, mtxShadow);

            colj = col_gray;
            coln = col_gray;
            colp = col_gray;
        }

        /* joint point */
        for (int i = 0; i < HAND_JOINT_NUM; i ++)
        {
            float vec[3] = {hand_draw.joint[i].x, hand_draw.joint[i].y, hand_draw.joint[i].z};

            if (!is_shadow)
            {
                if      (i >= 17) colj = col_violet;
                else if (i >= 13) colj = col_cyan;
                else if (i >=  9) colj = col_green;
                else if (i >=  5) colj = col_yellow;
                else              colj = col_red;
            }

            float rad = s_gui_prop.joint_radius;
            draw_sphere (mtxGlobal, vec, rad, colj, is_shadow);
        }

        /* joint node */
        float rad = s_gui_prop.bone_radius;
        render_3d_bone (mtxGlobal, &hand_draw, 0,  1, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, &hand_draw, 0, 17, coln, rad, is_shadow);

        render_3d_bone (mtxGlobal, &hand_draw,  1,  5, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, &hand_draw,  5,  9, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, &hand_draw,  9, 13, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, &hand_draw, 13, 17, coln, rad, is_shadow);

        for (int i = 0; i < 5; i ++)
        {
            int idx0 = 4 * i + 1;
            int idx1 = idx0 + 1;
            render_3d_bone (mtxGlobal, &hand_draw, idx0,  idx1  , coln, rad, is_shadow);
            render_3d_bone (mtxGlobal, &hand_draw, idx0+1,idx1+1, coln, rad, is_shadow);
            render_3d_bone (mtxGlobal, &hand_draw, idx0+2,idx1+2, coln, rad, is_shadow);
        }

        /* palm region */
        if (!is_shadow)
        {
            render_palm_tri (mtxGlobal, &hand_draw, 0,  1,  5, colp);
            render_palm_tri (mtxGlobal, &hand_draw, 0,  5,  9, colp);
            render_palm_tri (mtxGlobal, &hand_draw, 0,  9, 13, colp);
            render_palm_tri (mtxGlobal, &hand_draw, 0, 13, 17, colp);
        }
    }
}

static void
render_3d_scene (int ofstx, int ofsty,
                 hand_landmark_result_t  *landmark,
                 palm_detection_result_t *detection)
{
    float mtxGlobal[16], mtxTouch[16];
    float floor_size_x = 300.0f;
    float floor_size_y = 300.0f;
    float floor_size_z = 300.0f;

    get_touch_event_matrix (mtxTouch);

    /* background */
    matrix_identity (mtxGlobal);
    matrix_translate (mtxGlobal, 0, 0, -s_gui_prop.camera_pos_z);
    matrix_mult (mtxGlobal, mtxGlobal, mtxTouch);
    matrix_translate (mtxGlobal, 0, -s_gui_prop.pose_scale_y, 0);
    matrix_scale  (mtxGlobal, floor_size_x, floor_size_y, floor_size_z);
    matrix_translate (mtxGlobal, 0, 1.0, 0);
    draw_floor (mtxGlobal, floor_size_x/10, floor_size_y/10);

    for (int hand_id = 0; hand_id < detection->num; hand_id ++)
    {
        hand_landmark_result_t *hand_landmark = &landmark[hand_id];
        render_skelton_3d (ofstx, ofsty, hand_landmark, &detection->palms[hand_id]);
    }

    if (s_gui_prop.draw_axis)
    {
        /* (xyz)-AXIS */
        matrix_identity (mtxGlobal);
        matrix_translate (mtxGlobal, 0, 0, -s_gui_prop.camera_pos_z);
        matrix_mult (mtxGlobal, mtxGlobal, mtxTouch);
        for (int i = -1; i <= 1; i ++)
        {
            for (int j = -1; j <= 1; j ++)
            {
                float col_base[] = {0.1, 0.5, 0.5, 0.5};
                float dx = s_gui_prop.pose_scale_x;
                float dy = s_gui_prop.pose_scale_y;
                float dz = s_gui_prop.pose_scale_z;
                float rad = 1;

                {
                    float v0[3] = {-dx, i * dy, j * dz};
                    float v1[3] = { dx, i * dy, j * dz};
                    float col_red[] = {1.0, 0.0, 0.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_red : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                    draw_sphere (mtxGlobal, v1, rad, col, 0);
                }
                {
                    float v0[3] = {i * dx, -dy, j * dz};
                    float v1[3] = {i * dx,  dy, j * dz};
                    float col_green[] = {0.0, 1.0, 0.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_green : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                    draw_sphere (mtxGlobal, v1, rad, col, 0);
                }
                {
                    float v0[3] = {i * dx, j * dy, -dz};
                    float v1[3] = {i * dx, j * dy,  dz};
                    float col_blue[] = {0.0, 0.0, 1.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_blue : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                    draw_sphere (mtxGlobal, v1, rad, col, 0);
                }
            }
        }
    }
}


static void
render_cropped_hand_image (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh, palm_detection_result_t *detection, unsigned int hand_id)
{
    float texcoord[8];

    if (detection->num <= hand_id)
        return;

    palm_t *palm = &(detection->palms[hand_id]);
    float x0 = palm->hand_pos[0].x;
    float y0 = palm->hand_pos[0].y;
    float x1 = palm->hand_pos[1].x; //    0--------1
    float y1 = palm->hand_pos[1].y; //    |        |
    float x2 = palm->hand_pos[2].x; //    |        |
    float y2 = palm->hand_pos[2].y; //    3--------2
    float x3 = palm->hand_pos[3].x;
    float y3 = palm->hand_pos[3].y;
    texcoord[0] = x0;   texcoord[1] = y0;
    texcoord[2] = x3;   texcoord[3] = y3;
    texcoord[4] = x1;   texcoord[5] = y1;
    texcoord[6] = x2;   texcoord[7] = y2;

    draw_2d_texture_ex_texcoord (srctex, ofstx, ofsty, texw, texh, texcoord);
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

#if defined (USE_QUANT_TFLITE_MODEL)
    sprintf (strbuf, "MODEL_INTQUANT: ON ");
    draw_dbgstr_ex (strbuf, glctx.disp_w - 250, glctx.disp_h - 48, 1.0f, col_white, col_bg);
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
    s_gui_prop.frame_color[0] = 0.0f;
    s_gui_prop.frame_color[1] = 0.5f;
    s_gui_prop.frame_color[2] = 1.0f;
    s_gui_prop.frame_color[3] = 1.0f;
    s_gui_prop.pose_scale_x = 100.0f;
    s_gui_prop.pose_scale_y = 100.0f;
    s_gui_prop.pose_scale_z = 100.0f;
    s_gui_prop.camera_pos_z = 200.0f;
    s_gui_prop.joint_radius = 6.0f;
    s_gui_prop.bone_radius  = 2.0f;
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
    static double ttime[10] = {0}, interval, invoke_ms0 = 0, invoke_ms1 = 0;

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
        palm_detection_result_t palm_ret = {0};
        hand_landmark_result_t  hand_ret[MAX_PALM_NUM] = {};
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);

        /* --------------------------------------- *
         *  palm detection
         * --------------------------------------- */
        bool enable_palm_detect = true;
        if (enable_palm_detect)
        {
            feed_palm_detection_image (&srctex, win_w, win_h);

            ttime[2] = pmeter_get_time_ms ();
            invoke_palm_detection (&palm_ret, 0);
            ttime[3] = pmeter_get_time_ms ();
            invoke_ms0 = ttime[3] - ttime[2];
        }
        else
        {
            invoke_palm_detection (&palm_ret, 1);
        }

        /* --------------------------------------- *
         *  hand landmark
         * --------------------------------------- */
        invoke_ms1 = 0;
        for (int hand_id = 0; hand_id < palm_ret.num; hand_id ++)
        {
            feed_hand_landmark_image (&srctex, win_w, win_h, &palm_ret, hand_id);

            ttime[4] = pmeter_get_time_ms ();
            invoke_hand_landmark (&hand_ret[hand_id]);
            ttime[5] = pmeter_get_time_ms ();
            invoke_ms1 += ttime[5] - ttime[4];
        }

        /* --------------------------------------- *
         *  render scene  (right half)
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render_3d_scene (draw_x, draw_y, hand_ret, &palm_ret);

        /* --------------------------------------- *
         *  render scene (left half)
         * --------------------------------------- */

        float camx = 100;
        float camy = 100;
        float camw = 500;
        float camh = 500;
        /* visualize the hand pose estimation results. */
        draw_2d_texture_ex (&srctex, camx, camy, camw, camh, 0);

        for (int hand_id = 0; hand_id < palm_ret.num; hand_id ++)
        {
            palm_t *palm = &(palm_ret.palms[hand_id]);
            render_palm_region (camx, camy, camw, camh, palm);
            render_skelton_2d (camx, camy, camw, camh, palm, &hand_ret[hand_id]);
        }

        /* draw cropped image of the hand area */
        for (int hand_id = 0; hand_id < palm_ret.num; hand_id ++)
        {
            float w = 100;
            float h = 100;
            float x = camx + camw;
            float y = camy + h * hand_id;
            float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

            render_cropped_hand_image (&srctex, x, y, w, h, &palm_ret, hand_id);
            draw_2d_rect (x, y, w, h, col_white, 2.0f);
        }


        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        glViewport (0, 0, win_w, win_h);

        DrawTFLiteConfigInfo ();

        if (s_gui_prop.draw_pmeter)
        {
            draw_pmeter (0, 40);
        }

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite0 :%5.1f [ms]\nTFLite1 :%5.1f [ms]",
            interval, invoke_ms0, invoke_ms1);
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

    texture_2d_t tex_cube;
    LoadInputTexture (&tex_cube, (char *)"floortile.png");
    init_cube ((float)w / (float)h, tex_cube.texid);

    asset_read_file (m_app->activity->assetManager,
                    (char *)PALM_DETECTION_MODEL_PATH, m_palmdet_tflite_model_buf);

    asset_read_file (m_app->activity->assetManager,
                    (char *)HAND_LANDMARK_MODEL_PATH, m_landmark_tflite_model_buf);

    ret = init_tflite_hand_landmark (
        (const char *)m_palmdet_tflite_model_buf.data(),  m_palmdet_tflite_model_buf.size(),
        (const char *)m_landmark_tflite_model_buf.data(), m_landmark_tflite_model_buf.size());

    setup_imgui (w, h, &imgui_data);

    glctx.disp_w = w;
    glctx.disp_h = h;
    LoadInputTexture (&glctx.tex_static, (char *)"pakutaso_vsign.jpg");

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

