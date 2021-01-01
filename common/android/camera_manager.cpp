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
#include <utility>
#include <queue>
#include <unistd.h>
#include <cinttypes>
#include <camera/NdkCameraManager.h>
#include "camera_manager.h"
#include "util_debug.h"
#include "camera_utils.h"


NDKCamera::NDKCamera ()
    : mCameraManager(nullptr),
      mActiveCameraId(""),
      mSessionState(CaptureSessionState::MAX_STATE)
{
    mCameraIDMap.clear();
    mCameraManager = ACameraManager_create();
    ASSERT(mCameraManager, "Failed to create cameraManager");

#if 0
    PrintCameras (mCameraManager);
#endif

    /* Pick up a camera to use */
    EnumerateCamera ();
    SelectCameraFacing (CAMERA_FACING_BACK);
    ASSERT (mActiveCameraId.size(), "Unknown ActiveCameraIdx");

    CALL_CAMERA (ACameraManager_registerAvailabilityCallback (mCameraManager, GetManagerListener()));
}


bool
NDKCamera::MatchCaptureSizeRequest (int32_t *cam_width, int32_t *cam_height, int32_t *cam_format)
{
    ACameraMetadata *metadata;
    CALL_CAMERA (ACameraManager_getCameraCharacteristics (mCameraManager, mActiveCameraId.c_str(), &metadata));

    ACameraMetadata_const_entry entry;
    CALL_CAMERA (ACameraMetadata_getConstEntry (metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry));

    for (int i = 0; i < entry.count; i += 4) 
    {
        int32_t format = entry.data.i32[i + 0];
        int32_t width  = entry.data.i32[i + 1];
        int32_t height = entry.data.i32[i + 2];
        int32_t input  = entry.data.i32[i + 3];
        LOGI ("CAMERA[%3d/%3d] (%4d, %4d) input(%d) fmt(%08x:%s)\n",
            i, entry.count, width, height, input, format, GetFormatStr (format));
    }

    *cam_width  = 640;
    *cam_height = 480;
    *cam_format = AIMAGE_FORMAT_YUV_420_888;

    return true;
}

void
NDKCamera::CreateSession (ANativeWindow* previewWindow)
{
    /*
     *  Create camera device
     *  +-----------------+
     *  | [ACameraDevice] +----+---> OnDeviceStateChanges()
     *  |     mDevice     |    +---> OnDeviceErrorChanges()
     *  +-----------------+
     */
    LOGI ("open camera \"%s\"", mActiveCameraId.c_str());
    CALL_CAMERA (ACameraManager_openCamera (mCameraManager, mActiveCameraId.c_str(),
                                            GetDeviceListener(), &mDevice));


    /* use ANativeWindow as ACaptureSessionOutput */
    LOGI ("mImgReaderNativeWin: %p", previewWindow);
    mImgReaderNativeWin = previewWindow;
    ANativeWindow_acquire (mImgReaderNativeWin);    /* ref counter ++ */

    /*
     *  Create capture session
     *  +-----------------------------------------------------------+
     *  |                       mSession                            +----+---> OnSessionClosed()
     *  +-----------------------------------------------------------+    +---> OnSessionReady ()
     *  | [ACaptureSessionOutputContainer]  [ACaptureSessionOutput] |    +---> OnSessionActive()
     *  |            mOutputs ----------------> mImgReaderOutput    |
     *  |                                     mImgReaderNativeWin   |
     *  +-----------------------------------------------------------+
     */
    LOGI ("Create capture session");
    CALL_CAMERA (ACaptureSessionOutputContainer_create (&mOutputs));
    CALL_CAMERA (ACaptureSessionOutput_create (mImgReaderNativeWin, &mImgReaderOutput));
    CALL_CAMERA (ACaptureSessionOutputContainer_add (mOutputs, mImgReaderOutput));

    mSessionState = CaptureSessionState::READY;
    CALL_CAMERA (ACameraDevice_createCaptureSession (mDevice, mOutputs, GetSessionListener(), &mSession));

    /*
     *  Create capture request
     *    [TEMPLATE_RECORD ] stable frame rate is used, and post-processing is set for recording quality.
     *    [TEMPLATE_PREVIEW] high frame rate is given priority over the highest-quality post-processing.
     *
     *  +-----------------------------------------------+
     *  | [ACaptureRequest]       [ACameraOutputTarget] |
     *  |  mCaptureRequest ------> mReqImgReaderOutput  |
     *  |                          mImgReaderNativeWin  |
     *  +-----------------------------------------------+
     */
    LOGI ("Create capture request");
    ACameraDevice_request_template req_template = TEMPLATE_RECORD;
    CALL_CAMERA (ACameraDevice_createCaptureRequest (mDevice, req_template, &mCaptureRequest));

    CALL_CAMERA (ACameraOutputTarget_create (mImgReaderNativeWin, &mReqImgReaderOutput));
    CALL_CAMERA (ACaptureRequest_addTarget (mCaptureRequest, mReqImgReaderOutput));

    LOGI ("CreateSession() done.");
}

NDKCamera::~NDKCamera() {
    LOGI ("~NDKCamera()");

    /* Stop session if it is on: */
    if (mSessionState == CaptureSessionState::ACTIVE)
    {
        ACameraCaptureSession_stopRepeating (mSession);
    }
    ACameraCaptureSession_close (mSession);

    /* Destroy capture request */
    CALL_CAMERA (ACaptureRequest_removeTarget (mCaptureRequest, mReqImgReaderOutput));
    ACaptureRequest_free (mCaptureRequest);
    ACameraOutputTarget_free (mReqImgReaderOutput);

    /* Destroy capture session */
    CALL_CAMERA (ACaptureSessionOutputContainer_remove (mOutputs, mImgReaderOutput));
    ACaptureSessionOutput_free (mImgReaderOutput);
    ACaptureSessionOutputContainer_free (mOutputs);

    ANativeWindow_release (mImgReaderNativeWin);     /* ref counter -- */

    if (mDevice)
    {
        CALL_CAMERA (ACameraDevice_close (mDevice));
    }
    mCameraIDMap.clear();
    if (mCameraManager)
    {
        CALL_CAMERA (ACameraManager_unregisterAvailabilityCallback (mCameraManager, GetManagerListener()));
        ACameraManager_delete (mCameraManager);
        mCameraManager = nullptr;
    }
}


void
NDKCamera::EnumerateCamera ()
{
    /* Create a list of currently connected camera devices */
    ACameraIdList *cameraIds = nullptr;
    CALL_CAMERA (ACameraManager_getCameraIdList (mCameraManager, &cameraIds));

    for (int i = 0; i < cameraIds->numCameras; i++)
    {
        const char *id = cameraIds->cameraIds[i];
        LOGI ("CAMERA_ID[%d/%d] \"%s\"", i, cameraIds->numCameras, id);

        /* Query the capabilities of a camera device. */
        ACameraMetadata *metadataObj;
        CALL_CAMERA (ACameraManager_getCameraCharacteristics (mCameraManager, id, &metadataObj));

        /* List all the entry tags in input ACameraMetadata. */
        int32_t count = 0;
        const uint32_t *tags = nullptr;
        CALL_CAMERA (ACameraMetadata_getAllTags (metadataObj, &count, &tags));

        for (int tagIdx = 0; tagIdx < count; tagIdx++)
        {
            if (ACAMERA_LENS_FACING == tags[tagIdx])
            {
                ACameraMetadata_const_entry lensInfo = {0,};
                CALL_CAMERA (ACameraMetadata_getConstEntry (metadataObj, tags[tagIdx], &lensInfo));

                CameraId cam(id);
                cam.index   = i;
                cam.facing_ = static_cast<acamera_metadata_enum_android_lens_facing_t>(lensInfo.data.u8[0]);
                mCameraIDMap[cam.id_] = cam;
                break;
            }
        }
        ACameraMetadata_free(metadataObj);
    }

    ASSERT(mCameraIDMap.size(), "No Camera Available on the device");
    ACameraManager_deleteCameraIdList (cameraIds);
}

bool
NDKCamera::SelectCameraFacing (int req_facing)
{
    std::map<std::string, CameraId>::iterator it;
    for (it = mCameraIDMap.begin(); it != mCameraIDMap.end(); it++)
    {
        CameraId *cam = &it->second;
        LOGI ("CAMERA[%d] id: %s: facing=%d", cam->index, cam->id_.c_str(), cam->facing_);

        if ((req_facing == CAMERA_FACING_BACK  && cam->facing_ == ACAMERA_LENS_FACING_BACK) ||
            (req_facing == CAMERA_FACING_FRONT && cam->facing_ == ACAMERA_LENS_FACING_FRONT))
        {
            mActiveCameraId = cam->id_;
            return true;
        }
    }

    // if no match facing camera found, pick up the first one to use...
    if (mActiveCameraId.length() == 0)
    {
        mActiveCameraId = mCameraIDMap.begin()->second.id_;
    }
    return false;
}


/* Toggle preview start/stop */
void
NDKCamera::StartPreview (bool start)
{
    if (start)
    {
        CALL_CAMERA (ACameraCaptureSession_setRepeatingRequest(mSession, nullptr, 1,
                                                               &mCaptureRequest, nullptr));
    }
    else if (mSessionState == CaptureSessionState::ACTIVE)
    {
        CALL_CAMERA (ACameraCaptureSession_stopRepeating(mSession));
    }
}



/* ----------------------------------------------------------------- *
 *  Camera Manager Listener object
 *
 *      ACameraManager_AvailabilityCallbacks
 *          |
 *          +-- OnCameraAvailable()
 *          +-- OnCameraUnavailable()
 * ----------------------------------------------------------------- */
static void
OnCameraAvailable (void *ctx, const char *id)
{
    //reinterpret_cast<NDKCamera*>(ctx)->OnCameraStatusChanged (id, true);
}

static void
OnCameraUnavailable (void *ctx, const char *id)
{
    //reinterpret_cast<NDKCamera*>(ctx)->OnCameraStatusChanged (id, false);
}

void
NDKCamera::OnCameraStatusChanged(const char *id, bool available)
{
    LOGI ("[NDKCamera::OnCameraStatusChanged] id: %s: available: %d", id, available);
    //mCameraIDMap[std::string(id)].available_ = available;
}

ACameraManager_AvailabilityCallbacks *
NDKCamera::GetManagerListener() 
{
    static ACameraManager_AvailabilityCallbacks cameraMgrListener = {
        .context             = this,
        .onCameraAvailable   = ::OnCameraAvailable,
        .onCameraUnavailable = ::OnCameraUnavailable,
    };
    return &cameraMgrListener;
}


/* ----------------------------------------------------------------- *
 *  CameraDevice callbacks
 *
 *      ACameraDevice_stateCallbacks
 *          |
 *          +-- OnDeviceStateChanges()
 *          +-- OnDeviceErrorChanges()
 * ----------------------------------------------------------------- */
static void
OnDeviceStateChanges (void *ctx, ACameraDevice *dev)
{
    //reinterpret_cast<NDKCamera*>(ctx)->OnDeviceState (dev);
}

static void
OnDeviceErrorChanges (void *ctx, ACameraDevice *dev, int err)
{
    //reinterpret_cast<NDKCamera*>(ctx)->OnDeviceError (dev, err);
}

ACameraDevice_stateCallbacks *
NDKCamera::GetDeviceListener()
{
    static ACameraDevice_stateCallbacks cameraDeviceListener = {
        .context        = this,
        .onDisconnected = ::OnDeviceStateChanges,
        .onError        = ::OnDeviceErrorChanges,
    };
    return &cameraDeviceListener;
}

void
NDKCamera::OnDeviceState (ACameraDevice *dev)
{
    std::string id(ACameraDevice_getId(dev));
    LOGW("device %s is disconnected", id.c_str());

    mCameraIDMap[id].available_ = false;
    ACameraDevice_close (dev);
    mCameraIDMap.erase(id);
}

void
NDKCamera::OnDeviceError (ACameraDevice *dev, int err)
{
    std::string id(ACameraDevice_getId(dev));
    LOGI("CameraDevice %s is in error %#x", id.c_str(), err);

    CameraId& cam = mCameraIDMap[id];

    switch (err) {
    case ERROR_CAMERA_IN_USE:
        cam.available_ = false;
        break;
    case ERROR_CAMERA_SERVICE:
    case ERROR_CAMERA_DEVICE:
    case ERROR_CAMERA_DISABLED:
    case ERROR_MAX_CAMERAS_IN_USE:
        cam.available_ = false;
        break;
    default:
      LOGI("Unknown Camera Device Error: %#x", err);
    }
}


/* ----------------------------------------------------------------- *
 *  CaptureSession state callbacks
 *
 *      ACameraCaptureSession_stateCallbacks
 *          |
 *          +-- OnSessionClosed()
 *          +-- OnSessionReady()
 *          +-- OnSessionActive()
 * ----------------------------------------------------------------- */
static void
OnSessionClosed (void *ctx, ACameraCaptureSession *ses)
{
    LOGI ("OnSessionClosed (%p)", ses);
    //reinterpret_cast<NDKCamera*>(ctx)->OnSessionState (ses, CaptureSessionState::CLOSED);
}

static void
OnSessionReady (void *ctx, ACameraCaptureSession *ses)
{
    LOGI ("OnSessionReady (%p)", ses);
    //reinterpret_cast<NDKCamera*>(ctx)->OnSessionState (ses, CaptureSessionState::READY);
}

static void
OnSessionActive (void *ctx, ACameraCaptureSession *ses)
{
    LOGI ("OnSessionActive (%p)", ses);
    //reinterpret_cast<NDKCamera*>(ctx)->OnSessionState (ses, CaptureSessionState::ACTIVE);
}

ACameraCaptureSession_stateCallbacks *
NDKCamera::GetSessionListener()
{
    static ACameraCaptureSession_stateCallbacks sessionListener = {
        .context  = this,
        .onActive = ::OnSessionActive,
        .onReady  = ::OnSessionReady,
        .onClosed = ::OnSessionClosed,
    };
    return &sessionListener;
}

void 
NDKCamera::OnSessionState (ACameraCaptureSession *ses, CaptureSessionState state)
{
    if (!ses || ses != mSession)
    {
        LOGW ("CaptureSession is %s", (ses ? "NOT our session" : "NULL"));
        return;
    }
    ASSERT(state < CaptureSessionState::MAX_STATE, "Wrong state %d", state);

    mSessionState = state;
}




/* ----------------------------------------------------------------- *
 *  ImageReaderHelper for AHardwareBuffer
 * ----------------------------------------------------------------- */
ImageReaderHelper::ImageReaderHelper ()
{
    mFormat    = AIMAGE_FORMAT_PRIVATE;
    mUsage     = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
    mMaxImages = 3;
}

ImageReaderHelper::~ImageReaderHelper()
{
    mAcquiredImage.reset();
    if (mImgReaderNativeWin)
    {
        AImageReader_delete (mImgReader);
        // No need to call ANativeWindow_release on imageReaderAnw
    }
}


static void
OnImageAvailable (void *context, AImageReader *reader)
{
    ImageReaderHelper *thiz = reinterpret_cast<ImageReaderHelper *>(context);
    thiz->HandleImageAvailable();
}

void
ImageReaderHelper::HandleImageAvailable()
{
    std::lock_guard<std::mutex> lock(mMutex);

    mAvailableImages += 1; /* increment acquired image nums */
}


int
ImageReaderHelper::InitImageReader (int width, int height)
{
    LOGI ("InitImageReader(%d, %d)", width, height);

    mWidth  = width;
    mHeight = height;

    if (mImgReader != nullptr || mImgReaderNativeWin != nullptr)
    {
        ReleaseImageReader ();
    }

    /* AImageReader for GPU Reading */
    int ret = AImageReader_newWithUsage (mWidth, mHeight, mFormat, mUsage, mMaxImages, &mImgReader);
    if (ret != AMEDIA_OK || mImgReader == nullptr)
    {
        DBG_LOGE ("Failed to create new AImageReader");
        return -1;
    }

    /* Set Callback fuction which is called when a new image is available */
    mAvailableImages = 0;
    AImageReader_ImageListener readerAvailableCb {this, OnImageAvailable};
    media_status_t stat = AImageReader_setImageListener (mImgReader, &readerAvailableCb);
    if (stat != AMEDIA_OK)
    {
        DBG_LOGE ("Failed to set image available listener, ret=%d.", ret);
        return ret;
    }

    /* ANativeWindow */
    stat = AImageReader_getWindow (mImgReader, &mImgReaderNativeWin);
    if (stat != AMEDIA_OK || mImgReaderNativeWin == nullptr)
    {
        DBG_LOGE ("Failed to get ANativeWindow from AImageReader, ret=%d", ret);
        return -1;
    }

    return 0;
}

int
ImageReaderHelper::ReleaseImageReader ()
{
    LOGI ("ReleaseImageReader()");

    mAcquiredImage.reset();
    if (mImgReaderNativeWin)
    {
        AImageReader_delete (mImgReader);
    }
    mImgReader          = nullptr;
    mImgReaderNativeWin = nullptr;

    return 0;
}

int
ImageReaderHelper::GetBufferDimension (int *width, int *height)
{
    *width  = mWidth;
    *height = mHeight;
    return 0;
}

ANativeWindow *
ImageReaderHelper::GetNativeWindow()
{
    return mImgReaderNativeWin;
}


int
ImageReaderHelper::GetCurrentHWBuffer (AHardwareBuffer **outBuffer)
{
    std::lock_guard<std::mutex> lock(mMutex);

    int ret;
    if (mAvailableImages > 0)
    {
        mAvailableImages -= 1;  /* decrement acquired image nums */

        AImage *aimage = nullptr;
        ret = AImageReader_acquireLatestImage (mImgReader, &aimage);
        if (ret != AMEDIA_OK || aimage == nullptr)
        {
            DBG_LOGE("Failed to acquire image, ret=%d, outIamge=%p.", ret, aimage);
        }
        else
        {
            // Any exisitng in mAcquiredImage will be deleted and released automatically.
            mAcquiredImage.reset (aimage);
        }
    }

    if (mAcquiredImage == nullptr)
    {
        return -EAGAIN;
    }

    AHardwareBuffer *ahw_buffer = nullptr;
    ret = AImage_getHardwareBuffer (mAcquiredImage.get(), &ahw_buffer);
    if (ret != AMEDIA_OK || ahw_buffer == nullptr)
    {
        DBG_LOGE("Faild to get hardware buffer, ret=%d, outBuffer=%p.", ret, ahw_buffer);
        return -ENOMEM;
    }

#if 0
    AHardwareBuffer_Desc outDesc;
    AHardwareBuffer_describe (ahw_buffer, &outDesc);
    DBG_LOGI("[AHardwareBuffer] format: %d",  outDesc.format);
    DBG_LOGI("[AHardwareBuffer] width : %d",  outDesc.width);
    DBG_LOGI("[AHardwareBuffer] height: %d",  outDesc.height);
    DBG_LOGI("[AHardwareBuffer] layers: %d",  outDesc.layers);
    DBG_LOGI("[AHardwareBuffer] stride: %d",  outDesc.stride);
    DBG_LOGI("[AHardwareBuffer] usage : %lu", outDesc.usage);
#endif

    *outBuffer = ahw_buffer;
    return 0;
}


