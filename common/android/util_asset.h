#ifndef _UTIL_ASSET_H_
#define _UTIL_ASSET_H_

#include <vector>
#include <android/asset_manager.h>

bool    asset_read_file (AAssetManager *assetMgr, char *fname, std::vector<uint8_t>&buf);
uint8_t *asset_read_image (AAssetManager *assetMgr, char *fname, int32_t *img_w, int32_t *img_h);
void    asset_free_image (uint8_t *image_buf);

#endif /* _UTIL_ASSET_H_ */

