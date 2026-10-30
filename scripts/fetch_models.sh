#!/usr/bin/env bash
# Download third-party code and weights into vendor/ and models/ (both gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models vendor

[ -d vendor/SOME ] || git clone --depth 1 https://github.com/openvpi/SOME.git vendor/SOME
[ -d vendor/SOFA ] || git clone --depth 1 https://github.com/qiuqiao/SOFA.git vendor/SOFA
[ -d vendor/HubertFA ] || git clone --depth 1 --branch v0.0.7 https://github.com/wolfgitpr/HubertFA.git vendor/HubertFA

fetch_zip() { # url dest_dir
  [ -d "$2" ] && return
  mkdir -p "$2" && curl -fL -o "$2.zip" "$1" && unzip -q "$2.zip" -d "$2" && rm "$2.zip"
}
# Synthesis assets are supplied locally by the operator.
# Synthesis assets are supplied locally by the operator.
fetch_zip https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip models/some
if [ ! -f models/game/GAME-1.0.3-medium-onnx/estimator.onnx ]; then
  mkdir -p models/game
  curl -fL --retry 2 -o models/game/model.zip https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-medium-onnx.zip
  unzip -oq models/game/model.zip -d models/game
  rm models/game/model.zip
fi
fetch_zip https://github.com/qiuqiao/SOFA/releases/download/v1.0.1/pretrained_mandarin_singing.zip models/sofa_mandarin
if [ ! -f models/hubertfa/1218_hfa_model_new_dict/model.onnx ]; then
  mkdir -p models/hubertfa
  curl -fL --retry 2 -o models/hubertfa/model.zip https://github.com/wolfgitpr/HubertFA/releases/download/v0.0.7/1218_hfa_model_new_dict.zip
  unzip -oq models/hubertfa/model.zip -d models/hubertfa
  rm models/hubertfa/model.zip
fi
# Optional: RMVPE f0 (falls back to Praat without it). Must be yxlllc's build, not RVC's rmvpe.pt:
#   https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip -> models/rmvpe/model.pt
# audio-separator downloads its RoFormer checkpoints into models/separator on first use.
