/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAMERA_NATIVE_CAMERA_H
#define CAMERA_NATIVE_CAMERA_H
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraError.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraMetadataTags.h>
#include <media/NdkImageReader.h>
#include "util_debug.h"

enum class CaptureSessionState : int32_t {
    READY = 0,  // session is ready
    ACTIVE,     // session is busy
    CLOSED,     // session is closed(by itself or a new session evicts)
    MAX_STATE
};

#define CAMERA_FACING_BACK  0
#define CAMERA_FACING_FRONT 1

class CameraId;

class NDKCamera {
private:
    ACameraManager                  *mCameraManager;
    std::map<std::string, CameraId> mCameraIDMap;
    std::string                     mActiveCameraId;

    /* Camera device */
    ACameraDevice                   *mDevice {nullptr};

    ANativeWindow                   *mImgReaderNativeWin {nullptr};

    /* Capture session */
    ACaptureSessionOutputContainer  *mOutputs {nullptr};
    ACaptureSessionOutput           *mImgReaderOutput {nullptr};
    ACameraCaptureSession           *mSession {nullptr};
    CaptureSessionState             mSessionState;

    /* Capture request */
    ACaptureRequest                 *mCaptureRequest {nullptr};
    ACameraOutputTarget             *mReqImgReaderOutput {nullptr};

    ACameraManager_AvailabilityCallbacks    *GetManagerListener();
    ACameraDevice_stateCallbacks            *GetDeviceListener();
    ACameraCaptureSession_stateCallbacks    *GetSessionListener();

public:
    NDKCamera ();
    ~NDKCamera ();

    void EnumerateCamera (void);
    bool SelectCameraFacing (int req_facing);
    bool MatchCaptureSizeRequest (int32_t *cam_width, int32_t *cam_height, int32_t *cam_format);

    void CreateSession (ANativeWindow *previewWindow);

    void OnCameraStatusChanged (const char *id, bool available);
    void OnDeviceState (ACameraDevice *dev);
    void OnDeviceError (ACameraDevice *dev, int err);
    void OnSessionState (ACameraCaptureSession *ses, CaptureSessionState state);
    void StartPreview (bool start);
};

// helper classes to hold enumerated camera
class CameraId {
public:
    int             index;
    std::string     id_;
    acamera_metadata_enum_android_lens_facing_t facing_;
    bool            available_;  // free to use ( no other apps are using

    explicit CameraId(const char* id)
      : facing_(ACAMERA_LENS_FACING_FRONT),
        available_(false)
    {
        id_ = id;
    }

    explicit CameraId(void) { CameraId(""); }
};



/* ----------------------------------------------------------------- *
 *  ImageReaderHelper for AHardwareBuffer
 * ----------------------------------------------------------------- */
class ImageReaderHelper {
public:
    using ImagePtr = std::unique_ptr<AImage, decltype(&AImage_delete)>;

    ImageReaderHelper ();
    ~ImageReaderHelper ();

    int     InitImageReader (int width, int height);
    int     ReleaseImageReader ();
    void    HandleImageAvailable ();

    int     GetCurrentHWBuffer (AHardwareBuffer** outBuffer);
    int     GetBufferDimension (int *width, int *height);
    ANativeWindow *GetNativeWindow ();

private:
    int             mWidth, mHeight, mFormat, mMaxImages;
    uint64_t        mUsage;

    std::mutex      mMutex;

    size_t          mAvailableImages{0};
    ImagePtr        mAcquiredImage {nullptr, AImage_delete};

    AImageReader    *mImgReader {nullptr};
    ANativeWindow   *mImgReaderNativeWin {nullptr};
};

#endif  // CAMERA_NATIVE_CAMERA_H
