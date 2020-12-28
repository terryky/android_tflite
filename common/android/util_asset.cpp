/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <string>
#include <vector>
#include <android/asset_manager.h>
#include "util_asset.h"

//#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>



bool
asset_read_file (AAssetManager *assetMgr, char *fname, std::vector<uint8_t>&buf) 
{
    AAsset* assetDescriptor = AAssetManager_open(assetMgr, fname, AASSET_MODE_BUFFER);
    if (assetDescriptor == NULL)
    {
        return false;
    }

    size_t fileLength = AAsset_getLength(assetDescriptor);

    buf.resize(fileLength);
    int64_t readSize = AAsset_read(assetDescriptor, buf.data(), buf.size());

    AAsset_close(assetDescriptor);

    return (readSize == buf.size());
}

uint8_t *
asset_read_image (AAssetManager *assetMgr, char *fname, int32_t *img_w, int32_t *img_h)
{
    int32_t  width, height, channel_count;
    uint8_t* img_buf;
    bool     ret;

    /* read asset file */
    std::vector<uint8_t> read_buf;
    ret = asset_read_file (assetMgr, fname, read_buf);
    if (ret != true)
        return nullptr;

    /* decode image data to RGBA8888 */
    img_buf = stbi_load_from_memory (read_buf.data(), read_buf.size(),
            &width, &height, &channel_count, 4);

    *img_w = width;
    *img_h = height;

    return img_buf;
}

void
asset_free_image (uint8_t *image_buf)
{
    stbi_image_free (image_buf);
}

