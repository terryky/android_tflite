# GPU Accelerated TensorFlow Lite applications on Android NDK.
Run and measure the performance of TensorFlow Lite GPU Delegate on Android NDK.



## 1. Applications
| App name    | Descriptions |
|:-----------:|:------------:|
| [tflite_posenet](https://github.com/terryky/android_tflite/tree/master/tflite_posenet)| ![img](tflite_posenet/tflite_posenet.png " image") <br> Pose Estimation.|
| [tflite_style_transfer](https://github.com/terryky/android_tflite/tree/master/tflite_style_transfer)| ![img](tflite_style_transfer/style_transfer.png " image") <br> Artistic style transfer.|


## 2. How to Build & Run

### 2.1 setup environment

- Download and install [Android NDK](https://developer.android.com/ndk/downloads).

```
$ mkdir ~/Android/
$ mv ~/Download/android-ndk-r20b-linux-x86_64.zip ~/Android
$ cd ~/Android
$ unzip android-ndk-r20b-linux-x86_64.zip
```

- Download and install [bazel](https://docs.bazel.build/versions/master/install-ubuntu.html).

```
$ wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh
$ chmod 755 bazel-3.1.0-installer-linux-x86_64.sh
$ sudo ./bazel-3.1.0-installer-linux-x86_64.sh
```

### 2.2 build TensorFlow Lite library and GPU Delegate library

- run the build script to build TensorFlow Library

```
$ mkdir ~/work
$ git clone https://github.com/terryky/android_tflite.git
$ cd android_tflite/third_party/
$ ./build_libtflite_r2.4_android.sh

(Tensorflow configure will start after a while. Please enter according to your environment)


$ ls -l tensorflow/bazel-bin/tensorflow/lite/
-r-xr-xr-x  1 terryky terryky 3118552 Dec 26 19:58 libtensorflowlite.so*

$ ls -l tensorflow/bazel-bin/tensorflow/lite/delegates/gpu/
-r-xr-xr-x 1 terryky terryky 80389344 Dec 26 19:59 libtensorflowlite_gpu_delegate.so*
```




### 2.3 Build Android Applications
- Download and install [Android Studio](https://developer.android.com/studio/install).
- Start Android Studio.

```
$ cd ${ANDROID_STUDIO_INSTALL_DIR}/android-studio/bin/
$ ./studio.sh
```

- Install NDK 20.0 by SDK Manager of Android Studio.
- Open application folder (eg. ```~/work/android_tflite/tflite_posenet```).
- Build and Run.

## Tested Environment

| Host PC             | Target Device           |
|:-------------------:|:-----------------------:|
| x86_64              | arm64-v8a               |
| Ubuntu 18.04.4 LTS  | Android 9 (API Level 28)|
| Android NDK r20b    |                         |
