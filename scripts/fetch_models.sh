#!/usr/bin/env bash
# Download third-party code and weights into vendor/ and models/ (both gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models vendor

[ -d vendor/SOME ] || git clone --depth 1 https://github.com/openvpi/SOME.git vendor/SOME
[ -d vendor/SOFA ] || git clone --depth 1 https://github.com/qiuqiao/SOFA.git vendor/SOFA

fetch_zip() { # url dest_dir
  [ -d "$2" ] && return
  mkdir -p "$2" && curl -fL -o "$2.zip" "$1" && unzip -q "$2.zip" -d "$2" && rm "$2.zip"
}
# Synthesis assets are supplied locally by the operator.
# Synthesis assets are supplied locally by the operator.
fetch_zip https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip models/some
fetch_zip https://github.com/qiuqiao/SOFA/releases/download/v1.0.1/pretrained_mandarin_singing.zip models/sofa_mandarin
# Optional: RMVPE f0 (falls back to Praat without it). Must be yxlllc's build, not RVC's rmvpe.pt:
#   https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip -> models/rmvpe/model.pt
# audio-separator downloads its RoFormer checkpoints into models/separator on first use.
