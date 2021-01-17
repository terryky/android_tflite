#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1bW9Ip5kXNFcdvHC0dP2p7hSPw2JU1lyE" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1bW9Ip5kXNFcdvHC0dP2p7hSPw2JU1lyE" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
