package com.glesapp.glesapp;

import android.Manifest;
import android.app.NativeActivity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.PopupWindow;


public class GLESAppNativeActivity extends NativeActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {
    volatile GLESAppNativeActivity _savedInstance;
    PopupWindow _popupWindow;
    long[] _initParams;

    private final String DBG_TAG = "NDK-CAMERA";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        _savedInstance  = this;

        setImmersiveSticky();
        View decorView = getWindow().getDecorView();
        decorView.setOnSystemUiVisibilityChangeListener
                (new View.OnSystemUiVisibilityChangeListener() {
                    @Override
                    public void onSystemUiVisibilityChange(int visibility) {
                        setImmersiveSticky();
                    }
                });

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private boolean isCamera2Device() {
        CameraManager camMgr = (CameraManager)getSystemService(Context.CAMERA_SERVICE);
        boolean camera2Dev = true;
        try {
            String[] cameraIds = camMgr.getCameraIdList();
            Log.i(DBG_TAG, "-------------------- CHECK CAMERA2 AVAILABLE ------------------------");
            Log.i(DBG_TAG, "cameraIds.length = " + cameraIds.length);
            if (cameraIds.length != 0 ) {
                for (String id : cameraIds) {
                    CameraCharacteristics characteristics = camMgr.getCameraCharacteristics(id);
                    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
                    int facing      = characteristics.get(CameraCharacteristics.LENS_FACING);
                    Log.i(DBG_TAG, "CAM[" + id + "]");
                    Log.i(DBG_TAG, "  deviceLevel=" + deviceLevel);
                    Log.i(DBG_TAG, "  facing     =" + facing);
                    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
                        camera2Dev =  false;
                        Log.i(DBG_TAG, "  ##This Camera is LEGACY##");
                    }
                }
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
            camera2Dev = false;
        }
        return camera2Dev;
    }

    // get current rotation method
    int getRotationDegree() {
        return 90 * ((WindowManager)(getSystemService(WINDOW_SERVICE)))
                .getDefaultDisplay()
                .getRotation();
    }


    @Override
    protected void onResume() {
        super.onResume();
        setImmersiveSticky();
    }


    void setImmersiveSticky() {
        View decorView = getWindow().getDecorView();
        decorView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    @Override
    protected void onPause() {
        if (_popupWindow != null && _popupWindow.isShowing()) {
            _popupWindow.dismiss();
            _popupWindow = null;
        }
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    /* -------------------------------------------------------------------------- *
     * Request permission for CAMERA access.
     * -------------------------------------------------------------------------- */
    private static final int PERMISSION_REQUEST_CODE_CAMERA = 1;
    public void RequestCamera() {

        if(!isCamera2Device()) {
            Log.e(DBG_TAG, "Found legacy camera Device, this sample needs camera2 device");
            return;
        }

        String[] accessPermissions = new String[] {
            Manifest.permission.CAMERA,
        };

        /* check if already granted ? */
        boolean needRequire  = false;
        for(String access : accessPermissions) {
           int curPermission = ActivityCompat.checkSelfPermission(this, access);
           if(curPermission != PackageManager.PERMISSION_GRANTED) {
               needRequire = true; /* not yet granted */
               break;
           }
        }

        /* if it has not been granted yet, need to request permissions. */
        if (needRequire) {
            ActivityCompat.requestPermissions(
                    this, accessPermissions, PERMISSION_REQUEST_CODE_CAMERA);
            return;
        }

        /* if already granted, notify to Native App */
        notifyCameraPermission(true);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        /* if camera permission failed, the sample could not play */
        if (PERMISSION_REQUEST_CODE_CAMERA != requestCode) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }

        if(grantResults.length == 1) {
            boolean granted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
            notifyCameraPermission(granted);
        }
    }

    native static void notifyCameraPermission(boolean granted);

    static {
        System.loadLibrary("native-activity");
    }
}
