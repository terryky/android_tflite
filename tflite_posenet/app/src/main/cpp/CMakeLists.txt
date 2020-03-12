#
# Copyright (C) The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.4.1)

# build native_app_glue as a static lib
set(${CMAKE_C_FLAGS}, "${CMAKE_C_FLAGS}")
add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

# now build app's shared lib
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -Wall -Werror")

# Export ANativeActivity_onCreate(),
# Refer to: https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

get_filename_component(commonDir ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../common ABSOLUTE)
get_filename_component(thirdpDir ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party ABSOLUTE)
get_filename_component(jnilibDir ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs ABSOLUTE)

# download stb library
if ((NOT EXISTS ${thirdpDir}/stb) OR
    (NOT EXISTS ${thirdpDir}/stb/stb_image.h))
    execute_process(COMMAND git clone
                            https://github.com/nothings/stb.git
                            stb
                    WORKING_DIRECTORY ${commonDir})
endif()


# ------------------------------------------------------------
#  for TensorFlow lite 
# ------------------------------------------------------------

get_filename_component(tfliteDir ${thirdpDir}/tensorflow ABSOLUTE)

file(COPY ${tfliteDir}/bazel-genfiles/tensorflow/lite/libtensorflowlite.so
	DESTINATION ${jnilibDir}/arm64-v8a)

add_library(lib_tflite SHARED IMPORTED)
set_target_properties(lib_tflite PROPERTIES IMPORTED_LOCATION
	${jnilibDir}/arm64-v8a/libtensorflowlite.so)

include_directories(${tfliteDir}/
                    ${tfliteDir}/bazel-genfiles/../../../external/flatbuffers/include)





add_library(native-activity SHARED main.cpp
        tflite_posenet.cpp
        ${commonDir}/android/util_asset.cpp
        ${commonDir}/assertgl.c
        ${commonDir}/assertegl.c
        ${commonDir}/util_egl.c
        ${commonDir}/util_shader.c
        ${commonDir}/util_matrix.c
        ${commonDir}/util_texture.c
        ${commonDir}/util_render2d.c
        ${commonDir}/util_debugstr.c
        ${commonDir}/util_pmeter.c
        ${commonDir}/winsys/winsys_null.c)


include_directories(${thirdpDir}
                    ${commonDir}
                    ${commonDir}/android/
                    ${commonDir}/winsys/)

target_include_directories(native-activity PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue)

# add lib dependencies
target_link_libraries(native-activity
    android
    native_app_glue
    EGL
    GLESv2
    lib_tflite
    log)
