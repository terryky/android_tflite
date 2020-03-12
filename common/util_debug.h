#ifndef _UTIL_DEBUG_H_
#define _UTIL_DEBUG_H_

#include <stdio.h>
#include <stdlib.h> /* for exit() */



#include <android/log.h>
#define LOG_TAG "gles_app"
#define DBG_LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define DBG_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)




#define DBG_ASSERT(cond, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf (stderr, "ERROR(%s:%d) : ", __FILE__, __LINE__);    \
        fprintf (stderr,  __VA_ARGS__);                             \
        exit (-1);                                                  \
    } \
} while(0)



#endif /* _UTIL_DEBUG_H_ */