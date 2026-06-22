#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/LoveDA}"
CONNECTIONS="${LOVEDA_CONNECTIONS:-16}"

mkdir -p "$OUT_DIR"

download_segmented() {
  local url="$1"
  local name="$2"
  local out="$OUT_DIR/$name"
  local parts_dir="$out.parts"

  local size
  size="$(curl -fsSLI "$url" | awk 'tolower($1) == "content-length:" {print $2}' | tr -d '\r' | tail -1)"
  if [[ -z "$size" ]]; then
    echo "Could not determine size for $name" >&2
    return 1
  fi

  if [[ -f "$out" ]]; then
    local current
    current="$(stat -c '%s' "$out")"
    if [[ "$current" == "$size" ]]; then
      echo "$name already complete"
      return 0
    fi
  fi

  mkdir -p "$parts_dir"
  local chunk=$(( (size + CONNECTIONS - 1) / CONNECTIONS ))

  echo "Downloading $name ($size bytes) with $CONNECTIONS connections"
  for ((i = 0; i < CONNECTIONS; i++)); do
    local start=$(( i * chunk ))
    local end=$(( start + chunk - 1 ))
    if (( start >= size )); then
      break
    fi
    if (( end >= size )); then
      end=$(( size - 1 ))
    fi

    local part="$parts_dir/part-$(printf '%03d' "$i")"
    local expected=$(( end - start + 1 ))
    if [[ -f "$part" && "$(stat -c '%s' "$part")" == "$expected" ]]; then
      continue
    fi

    curl -fsSL --retry 5 --retry-delay 5 -r "$start-$end" -o "$part" "$url" &
  done
  wait

  for ((i = 0; i < CONNECTIONS; i++)); do
    local start=$(( i * chunk ))
    if (( start >= size )); then
      break
    fi
    local part="$parts_dir/part-$(printf '%03d' "$i")"
    [[ -s "$part" ]] || { echo "Missing $part" >&2; return 1; }
  done

  cat "$parts_dir"/part-* > "$out"
  local final_size
  final_size="$(stat -c '%s' "$out")"
  [[ "$final_size" == "$size" ]] || {
    echo "Size check failed for $name: got $final_size, expected $size" >&2
    return 1
  }
  echo "Completed $name"
}

download_plain() {
  local url="$1"
  local name="$2"
  curl -fsSL --retry 5 --retry-delay 5 -o "$OUT_DIR/$name" "$url"
}

download_plain "https://zenodo.org/records/5706578/files/Datasheet.pdf?download=1" "Datasheet.pdf"
download_segmented "https://zenodo.org/records/5706578/files/Train.zip?download=1" "Train.zip"
download_segmented "https://zenodo.org/records/5706578/files/Val.zip?download=1" "Val.zip"
download_segmented "https://zenodo.org/records/5706578/files/Test.zip?download=1" "Test.zip"
