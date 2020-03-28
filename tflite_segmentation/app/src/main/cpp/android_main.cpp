/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <android_native_app_glue.h>
#include "app_engine.h"


static AppEngine *s_pEngineObj = nullptr;

AppEngine *
GetAppEngine(void)
{
    return s_pEngineObj;
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
