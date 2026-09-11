#!/usr/bin/env bash
set -euo pipefail

RAW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"

fetch() {
  mkdir -p "$RAW/$1"
  curl -sL -o "$RAW/$1/$2" "https://github.com/mggg-states/$1-shapefiles/raw/master/$2"
  unzip -o -q "$RAW/$1/$2" -d "$RAW/$1"
}

fetch PA PA.zip
fetch NC NC_VTD.zip
fetch MD MD_precincts.zip
echo "shapefiles in $RAW/{PA,NC,MD}"
