/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <android_native_app_glue.h>
#include "app_engine.h"
#include "gestureDetector.h"


static AppEngine *s_pEngineObj = nullptr;

AppEngine *
GetAppEngine(void)
{
    return s_pEngineObj;
}


static void
handle_imgui_input (AppEngine *engine, AInputEvent *event)
{
    int32_t  action = AMotionEvent_getAction (event);
    int32_t  index  = (action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    uint32_t flags  = (action & AMOTION_EVENT_ACTION_MASK);
    int32_t  count  = AMotionEvent_getPointerCount (event);
    int32_t released_pointer_id;

    float x = AMotionEvent_getX (event, index);
    float y = AMotionEvent_getY (event, index);

    switch (flags) {
    case AMOTION_EVENT_ACTION_DOWN:
        engine->button_cb (index, 1, x, y);
        engine->mousemove_cb (x, y);
        break;

    case AMOTION_EVENT_ACTION_POINTER_DOWN:
        break;

    case AMOTION_EVENT_ACTION_UP:
        engine->button_cb (index, 0, x, y);
        break;

    case AMOTION_EVENT_ACTION_POINTER_UP:
        released_pointer_id = AMotionEvent_getPointerId (event, index);
        x = AMotionEvent_getX (event, released_pointer_id);
        y = AMotionEvent_getY (event, released_pointer_id);
        break;

    case AMOTION_EVENT_ACTION_MOVE:
        engine->mousemove_cb (x, y);
        break;

    case AMOTION_EVENT_ACTION_CANCEL:
        break;

    default:
        break;
    }
}

static int32_t
engine_handle_input (struct android_app *app, AInputEvent *event)
{
    AppEngine* engine = reinterpret_cast<AppEngine*>(app->userData);

    if (AInputEvent_getType (event) != AINPUT_EVENT_TYPE_MOTION)
        return 0;

    ndk_helper::GESTURE_STATE tapState  = engine->tap_detector_ .Detect(event);
    ndk_helper::GESTURE_STATE dragState = engine->drag_detector_.Detect(event);

    handle_imgui_input (engine, event);

    return 0;
}

static void
ProcessAndroidCmd (struct android_app* app, int32_t cmd) 
{
    AppEngine* engine = reinterpret_cast<AppEngine*>(app->userData);

    switch (cmd) {
    // The window is being shown, get it ready.
    case APP_CMD_INIT_WINDOW: 
        if (engine->AndroidApp()->window != NULL) 
        {
            engine->OnAppInitWindow();
        }
        break;

    // The window is being hidden or closed, clean it up.
    case APP_CMD_TERM_WINDOW: 
        engine->OnAppTermWindow();
        break;
    }
}


/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
void android_main(struct android_app* state) 
{
    AppEngine engine(state);
    state->userData = reinterpret_cast<void*>(&engine);
    state->onAppCmd = ProcessAndroidCmd;
    state->onInputEvent = engine_handle_input;

    s_pEngineObj = &engine;

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
                engine.DeleteCamera();
                s_pEngineObj = nullptr;
                return;
            }
        }

        engine.UpdateFrame ();
    }
}
