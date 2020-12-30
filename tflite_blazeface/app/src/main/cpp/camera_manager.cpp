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

NDKCamera::NDKCamera()
    : cameraMgr_(nullptr),
      activeCameraId_(""),
      outputContainer_(nullptr),
      captureSessionState_(CaptureSessionState::MAX_STATE)
{
    valid_ = false;

    requests_.resize(CAPTURE_REQUEST_COUNT);
    memset(requests_.data(), 0, requests_.size() * sizeof(requests_[0]));

    cameras_.clear();
    cameraMgr_ = ACameraManager_create();
    ASSERT(cameraMgr_, "Failed to create cameraManager");

    // Pick up a camera to use
    EnumerateCamera();
    ASSERT(activeCameraId_.size(), "Unknown ActiveCameraIdx");

    // Create back facing camera device
    CALL_MGR(openCamera(cameraMgr_, activeCameraId_.c_str(), GetDeviceListener(),
                        &cameras_[activeCameraId_].device_));

    CALL_MGR(registerAvailabilityCallback(cameraMgr_, GetManagerListener()));

    valid_ = true;
}


bool NDKCamera::MatchCaptureSizeRequest(int32_t *cam_width, int32_t *cam_height, int32_t *cam_format)
{
    ACameraMetadata* metadata;
    CALL_MGR(getCameraCharacteristics(cameraMgr_, activeCameraId_.c_str(), &metadata));

    ACameraMetadata_const_entry entry;
    CALL_METADATA(getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry));

    for (int i = 0; i < entry.count; i += 4) 
    {
        int32_t format = entry.data.i32[i + 0];
        int32_t width  = entry.data.i32[i + 1];
        int32_t height = entry.data.i32[i + 2];
        int32_t input  = entry.data.i32[i + 3];
        LOGI ("CAMERA[%3d/%3d] (%4d, %4d) fmt(%08x) input(%d)\n", i, entry.count, width, height, format, input);
    }

    *cam_width  = 640;
    *cam_height = 480;
    *cam_format = AIMAGE_FORMAT_YUV_420_888;

    return true;
}

void
NDKCamera::CreateSession(ANativeWindow* previewWindow) 
{
    // Create output from this app's ANativeWindow, and add into output container
    requests_[PREVIEW_REQUEST_IDX].outputNativeWindow_ = previewWindow;
    requests_[PREVIEW_REQUEST_IDX].template_           = TEMPLATE_PREVIEW;

    CALL_CONTAINER(create(&outputContainer_));

    for (auto& req : requests_)
    {
        ANativeWindow_acquire(req.outputNativeWindow_);
        CALL_OUTPUT(create(req.outputNativeWindow_, &req.sessionOutput_));
        CALL_CONTAINER(add(outputContainer_, req.sessionOutput_));
        CALL_TARGET(create(req.outputNativeWindow_, &req.target_));
        CALL_DEV(createCaptureRequest(cameras_[activeCameraId_].device_, req.template_, &req.request_));
        CALL_REQUEST(addTarget(req.request_, req.target_));
    }

    // Create a capture session for the given preview request
    captureSessionState_ = CaptureSessionState::READY;
    CALL_DEV(createCaptureSession(cameras_[activeCameraId_].device_,
                                  outputContainer_, GetSessionListener(), &captureSession_));
}

NDKCamera::~NDKCamera() {
    valid_ = false;

    // stop session if it is on:
    if (captureSessionState_ == CaptureSessionState::ACTIVE) 
    {
        ACameraCaptureSession_stopRepeating(captureSession_);
    }
    ACameraCaptureSession_close(captureSession_);

    for (auto& req : requests_) 
    {
        CALL_REQUEST(removeTarget(req.request_, req.target_));
        ACaptureRequest_free(req.request_);
        ACameraOutputTarget_free(req.target_);

        CALL_CONTAINER(remove(outputContainer_, req.sessionOutput_));
        ACaptureSessionOutput_free(req.sessionOutput_);

        ANativeWindow_release(req.outputNativeWindow_);
    }

    requests_.resize(0);
    ACaptureSessionOutputContainer_free(outputContainer_);

    for (auto& cam : cameras_) 
    {
        if (cam.second.device_) 
        {
            CALL_DEV(close(cam.second.device_));
        }
    }
    cameras_.clear();
    if (cameraMgr_) 
    {
        CALL_MGR(unregisterAvailabilityCallback(cameraMgr_, GetManagerListener()));
        ACameraManager_delete(cameraMgr_);
        cameraMgr_ = nullptr;
    }
}


void
NDKCamera::EnumerateCamera() 
{
    ACameraIdList* cameraIds = nullptr;
    CALL_MGR(getCameraIdList(cameraMgr_, &cameraIds));

    for (int i = 0; i < cameraIds->numCameras; ++i) 
    {
        const char* id = cameraIds->cameraIds[i];

        ACameraMetadata* metadataObj;
        CALL_MGR(getCameraCharacteristics(cameraMgr_, id, &metadataObj));

        int32_t count = 0;
        const uint32_t* tags = nullptr;
        ACameraMetadata_getAllTags(metadataObj, &count, &tags);

        for (int tagIdx = 0; tagIdx < count; ++tagIdx) 
        {
            if (ACAMERA_LENS_FACING == tags[tagIdx]) 
            {
                ACameraMetadata_const_entry lensInfo = {0,};

                CALL_METADATA(getConstEntry(metadataObj, tags[tagIdx], &lensInfo));
                CameraId cam(id);
                cam.facing_ = static_cast<acamera_metadata_enum_android_lens_facing_t>(lensInfo.data.u8[0]);
                cam.owner_  = false;
                cam.device_ = nullptr;
                cameras_[cam.id_] = cam;

                /* select BACK_FACING camera */
                if (cam.facing_ == ACAMERA_LENS_FACING_BACK) 
                {
                    activeCameraId_ = cam.id_;
                }
                break;
            }
        }
        ACameraMetadata_free(metadataObj);
    }

    ASSERT(cameras_.size(), "No Camera Available on the device");

    // if no back facing camera found, pick up the first one to use...
    if (activeCameraId_.length() == 0) 
    {
        activeCameraId_ = cameras_.begin()->second.id_;
    }
    ACameraManager_deleteCameraIdList(cameraIds);
}


/* Toggle preview start/stop */
void
NDKCamera::StartPreview(bool start) 
{
    if (start) 
    {
        CALL_SESSION(setRepeatingRequest(captureSession_, nullptr, 1,
                                 &requests_[PREVIEW_REQUEST_IDX].request_, nullptr));
    }
    else if (captureSessionState_ == CaptureSessionState::ACTIVE) 
    {
        ACameraCaptureSession_stopRepeating(captureSession_);
    }
}



/* ----------------------------------------------------------------- *
 *  Camera Manager Listener object
 * ----------------------------------------------------------------- */
void OnCameraAvailable(void* ctx, const char* id) 
{
    reinterpret_cast<NDKCamera*>(ctx)->OnCameraStatusChanged(id, true);
}

void OnCameraUnavailable(void* ctx, const char* id) 
{
    reinterpret_cast<NDKCamera*>(ctx)->OnCameraStatusChanged(id, false);
}

void NDKCamera::OnCameraStatusChanged(const char* id, bool available) 
{
    if (valid_) 
    {
        cameras_[std::string(id)].available_ = available ? true : false;
    }
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
 * ----------------------------------------------------------------- */
void OnDeviceStateChanges(void* ctx, ACameraDevice* dev) 
{
    reinterpret_cast<NDKCamera*>(ctx)->OnDeviceState(dev);
}

void OnDeviceErrorChanges(void* ctx, ACameraDevice* dev, int err) 
{
    reinterpret_cast<NDKCamera*>(ctx)->OnDeviceError(dev, err);
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
NDKCamera::OnDeviceState(ACameraDevice* dev) 
{
    std::string id(ACameraDevice_getId(dev));
    LOGW("device %s is disconnected", id.c_str());

    cameras_[id].available_ = false;
    ACameraDevice_close(cameras_[id].device_);
    cameras_.erase(id);
}

void
NDKCamera::OnDeviceError(ACameraDevice* dev, int err) 
{
    std::string id(ACameraDevice_getId(dev));
    LOGI("CameraDevice %s is in error %#x", id.c_str(), err);

    CameraId& cam = cameras_[id];

    switch (err) {
    case ERROR_CAMERA_IN_USE:
        cam.available_ = false;
        cam.owner_     = false;
        break;
    case ERROR_CAMERA_SERVICE:
    case ERROR_CAMERA_DEVICE:
    case ERROR_CAMERA_DISABLED:
    case ERROR_MAX_CAMERAS_IN_USE:
        cam.available_ = false;
        cam.owner_     = false;
        break;
    default:
      LOGI("Unknown Camera Device Error: %#x", err);
    }
}


/* ----------------------------------------------------------------- *
 *  CaptureSession state callbacks
 * ----------------------------------------------------------------- */
void OnSessionClosed(void* ctx, ACameraCaptureSession* ses) 
{
    LOGW("session %p closed", ses);
    reinterpret_cast<NDKCamera*>(ctx)->OnSessionState(ses, CaptureSessionState::CLOSED);
}

void OnSessionReady(void* ctx, ACameraCaptureSession* ses) 
{
    LOGW("session %p ready", ses);
    reinterpret_cast<NDKCamera*>(ctx)->OnSessionState(ses, CaptureSessionState::READY);
}

void OnSessionActive(void* ctx, ACameraCaptureSession* ses) 
{
    LOGW("session %p active", ses);
    reinterpret_cast<NDKCamera*>(ctx)->OnSessionState(ses, CaptureSessionState::ACTIVE);
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
NDKCamera::OnSessionState(ACameraCaptureSession* ses, CaptureSessionState state) 
{
    if (!ses || ses != captureSession_) 
    {
        LOGW("CaptureSession is %s", (ses ? "NOT our session" : "NULL"));
        return;
    }

    ASSERT(state < CaptureSessionState::MAX_STATE, "Wrong state %d", state);

    captureSessionState_ = state;
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
    if (mImgReaderAnw)
    {
        AImageReader_delete (mImgReader);
        // No need to call ANativeWindow_release on imageReaderAnw
    }
}


static void
OnImageAvailable (void* obj, AImageReader*)
{
    ImageReaderHelper *thiz = reinterpret_cast<ImageReaderHelper *>(obj);
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
    mWidth  = width;
    mHeight = height;

    if (mImgReader != nullptr || mImgReaderAnw != nullptr)
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

    /* Set Callback fuction */
    AImageReader_ImageListener readerAvailableCb {this, OnImageAvailable};
    media_status_t stat = AImageReader_setImageListener (mImgReader, &readerAvailableCb);
    if (stat != AMEDIA_OK)
    {
        DBG_LOGE ("Failed to set image available listener, ret=%d.", ret);
        return ret;
    }

    /* ANativeWindow */
    stat = AImageReader_getWindow (mImgReader, &mImgReaderAnw);
    if (stat != AMEDIA_OK || mImgReaderAnw == nullptr)
    {
        DBG_LOGE ("Failed to get ANativeWindow from AImageReader, ret=%d", ret);
        return -1;
    }

    return 0;
}

int
ImageReaderHelper::ReleaseImageReader ()
{
    mAcquiredImage.reset();
    if (mImgReaderAnw)
    {
        AImageReader_delete (mImgReader);
    }
    mImgReader    = nullptr;
    mImgReaderAnw = nullptr;

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
    return mImgReaderAnw;
}


int
ImageReaderHelper::GetCurrentHWBuffer (AHardwareBuffer **outBuffer)
{
    std::lock_guard<std::mutex> lock(mMutex);

    int ret;
    if (mAvailableImages > 0)
    {
        AImage *outImage = nullptr;

        mAvailableImages -= 1;  /* decrement acquired image nums */

        ret = AImageReader_acquireLatestImage (mImgReader, &outImage);
        if (ret != AMEDIA_OK || outImage == nullptr)
        {
            DBG_LOGE("Failed to acquire image, ret=%d, outIamge=%p.", ret, outImage);
        }
        else
        {
            // Any exisitng in mAcquiredImage will be deleted and released automatically.
            mAcquiredImage.reset (outImage);
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


