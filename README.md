# android_gles_app
Android applications using OpenGLES graphics.



## applications
| App name    | Descriptions |
|:-----------:|:------------:|
| [tflite_posenet](https://github.com/terryky/android_tflite/tree/master/tflite_posenet)| ![img](tflite_posenet/tflite_posenet.png " image") <br> Pose Estimation.|


## How to Build & Run

### (step-1) Build TensorFlow Lite libraries.
- Download and install [Android NDK](https://developer.android.com/ndk/downloads).
- Download and install [bazel](https://docs.bazel.build/versions/master/install-ubuntu.html).
- Run a build script.


```
$ cd android_tflite/third_party/
$ ./build_libtflite_r2.0_android.sh
```

And then, TensorFlow Lite libraries are generated as follows.

```
$ cd android_tflite/third_party/tensorflow
$ ls -l bazel-genfiles/tensorflow/lite/
-r-xr-xr-x  1 terryky terryky 2166728 Mar 13 00:55 libtensorflowlite.so*

$ ls -l bazel-genfiles/tensorflow/lite/delegates/gpu
-r-xr-xr-x 1 terryky terryky 36963952 Mar 13 00:56 libtensorflowlite_gpu_gl.so*
```




### (step-2) Build Android Applications
- Download and install [Android Studio](https://developer.android.com/studio/install).
- Start Android Studio.
- Open application folder (eg. ```tflite_posenet```).
- Build and Run.

## Tested Envronment

| Host PC             | Target Device           |
|:-------------------:|:-----------------------:|
| x86_64              | arm64-v8a               |
| Ubuntu 18.04.4 LTS  | Android 9 (API Level 28)|
| Android NDK r20b    |                         |
