# GPU Accelerated TensorFlow Lite applications on Android NDK.
Run and measure the performance of TensorFlow Lite GPU Delegate on Android NDK.



## 1. Applications

### [Blazeface](https://github.com/terryky/android_tflite/tree/master/tflite_blazeface)
- Lightweight Face Detection.<br>
[<img src="tflite_blazeface/tflite_blazeface.png" width=500>](https://github.com/terryky/android_tflite/tree/master/tflite_blazeface)

### [Object Detection](https://github.com/terryky/android_tflite/tree/master/tflite_detection)
- Object Detection using MobileNet SSD.<br>
[<img src="tflite_detection/tflite_detection.png" width=500>](https://github.com/terryky/android_tflite/tree/master/tflite_detection)

### [Posenet](https://github.com/terryky/android_tflite/tree/master/tflite_posenet)
- Pose Estimation.<br>
[<img src="tflite_posenet/tflite_posenet.png" width=500>](https://github.com/terryky/android_tflite/tree/master/tflite_posenet)

### [Semantic Segmentation](https://github.com/terryky/android_tflite/tree/master/tflite_segmentation)
- Assign semantic labels to every pixel in the input image.<br>
[<img src="tflite_segmentation/tflite_segmentation.png" width=600>](https://github.com/terryky/android_tflite/tree/master/tflite_segmentation)

### [Artistic Style Transfer](https://github.com/terryky/android_tflite/tree/master/tflite_style_transfer)
- Create new artworks in artistic style.<br>
[<img src="tflite_style_transfer/style_transfer.png" width=600>](https://github.com/terryky/android_tflite/tree/master/tflite_style_transfer)



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
