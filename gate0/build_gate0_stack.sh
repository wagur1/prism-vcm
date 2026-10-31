#!/usr/bin/env bash
# Ubuntu/Kaggle only. All installs are local except missing apt dependencies.
set -Eeuo pipefail
export LC_ALL=C
GATE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$GATE/.." && pwd)
LOG="$GATE/build_log.txt"
STEP=initialization
fail() {
    printf '[GATE0 FAIL] step=%s exit=%s line=%s\n' "$STEP" "$1" "$2"
    rm -f -- "$GATE/build-pin.json" "$GATE/stock-pin.json" "$GATE/stack.json" || true
    exit "$1"
}
trap 'fail "$?" "$LINENO"' ERR
step() { STEP=$1; printf '\n[GATE0 STEP] %s\n' "$STEP"; }
die() { printf '%s\n' "$*"; fail 1 "$LINENO"; }

step initialize_build_log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
step initialize_workspace
mkdir -p "$GATE/build" "$GATE/src"
step prerequisites
[[ $(uname -s) == Linux ]] || die 'This builder requires Ubuntu Linux (Kaggle).'
PYTHON=${GATE0_PYTHON:-python3}
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
JOBS=${GATE0_JOBS:-$(nproc)}
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die 'GATE0_JOBS must be a positive integer.'
if (( JOBS > 4 )); then JOBS=4; fi
missing=()
for entry in cmake:cmake nasm:nasm git:git patch:patch gcc:gcc g++:g++ make:make ffmpeg:ffmpeg; do
    command -v "${entry#*:}" >/dev/null || missing+=("${entry%%:*}")
done
if ! dpkg-query -W -f='${Status}' libnuma-dev 2>/dev/null | grep -qx 'install ok installed'; then
    missing+=(libnuma-dev)
fi
if ((${#missing[@]})); then
    step apt_dependencies
    apt_runner=()
    if (( EUID != 0 )); then apt_runner=(sudo -n); fi
    "${apt_runner[@]}" apt-get update
    "${apt_runner[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
fi
step python_dependencies
export PYTHONPATH="$GATE/build/python-deps:$REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON" -c 'import numpy' ; then
    "$PYTHON" -m pip install --disable-pip-version-check --target "$GATE/build/python-deps" 'numpy>=1.24,<3'
fi
gcc --version
cmake --version
nasm -v
ffmpeg -version

# Old output pins must never survive a failed new build as current evidence.
rm -f -- "$GATE/build-pin.json" "$GATE/stock-pin.json" "$GATE/stack.json"
X265_URL=https://bitbucket.org/multicoreware/x265_git.git
HM_URL=https://vcgit.hhi.fraunhofer.de/jvet/HM.git
checkout_tag() {
    local url=$1 tag=$2 dest=$3
    if [[ ! -d "$dest/.git" ]]; then
        [[ ! -e "$dest" ]] || die "Refusing non-git source directory: $dest"
        git -c credential.helper= clone --depth 1 --branch "$tag" --single-branch "$url" "$dest"
    fi
    local expected actual
    expected=$(git -C "$dest" rev-parse "refs/tags/$tag^{commit}")
    actual=$(git -C "$dest" rev-parse HEAD)
    [[ $actual == "$expected" ]] || die "$dest HEAD is not tag $tag; preserve/remove that checkout explicitly."
    printf 'SOURCE %s tag=%s commit=%s\n' "$url" "$tag" "$actual"
}
export GIT_TERMINAL_PROMPT=0
step clone_x265_3.5
checkout_tag "$X265_URL" 3.5 "$GATE/src/x265"
step clone_stock_x265_3.5
# Clone the local git object database, never the patched working tree.
if [[ ! -e "$GATE/src/x265-stock" ]]; then
    git clone --no-hardlinks --branch 3.5 "$GATE/src/x265" "$GATE/src/x265-stock"
fi
checkout_tag "$X265_URL" 3.5 "$GATE/src/x265-stock"
git -C "$GATE/src/x265-stock" diff --exit-code HEAD

step apply_x265_residual_patch
PATCH="$GATE/x265_residual_patch.patch"
if git -C "$GATE/src/x265" apply --reverse --check "$PATCH"; then
    printf 'Patch already applied; retaining the source tree.\n'
else
    git -C "$GATE/src/x265" diff --exit-code HEAD || die 'Source has unexpected/partial edits; inspect build_log.txt before restoring it.'
    if git -C "$GATE/src/x265" apply --check "$PATCH" && git -C "$GATE/src/x265" apply "$PATCH"; then
        printf 'git apply succeeded.\n'
    else
        printf 'git apply failed; trying patch -p1 --forward. Hunk diagnostics follow.\n'
        if (cd "$GATE/src/x265" && patch -p1 --forward --batch --verbose --fuzz=0 \
                --reject-file="$GATE/patch-rejects.diff" < "$PATCH"); then
            printf 'patch fallback succeeded; verify any reported line offsets in PATCH_AUDIT.md.\n'
        else
            printf 'Rejected hunks (file names and hunk line numbers are also logged above):\n'
            if [[ -f "$GATE/patch-rejects.diff" ]]; then cat "$GATE/patch-rejects.diff"; fi
            die 'Patch failed. Partial edits/rejects are preserved for review; no build pin was emitted.'
        fi
    fi
fi

build_x265() {
    local src=$1 build=$2 shim=$3 label=$4
    step "configure_${label}_x265"
    cmake -S "$src/source" -B "$build" -G 'Unix Makefiles' \
        -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$build/install" \
        -DLIB_INSTALL_DIR=lib -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DHIGH_BIT_DEPTH=OFF -DMAIN12=OFF \
        -DLINKED_10BIT=OFF -DLINKED_12BIT=OFF \
        -DENABLE_SHARED=ON -DENABLE_CLI=OFF -DENABLE_PIC=ON \
        -DEXPORT_C_API=ON -DENABLE_ASSEMBLY=ON -DENABLE_LIBNUMA=ON
    step "build_${label}_x265"
    cmake --build "$build" --parallel "$JOBS"
    cmake --install "$build"
    [[ -f "$build/install/lib/libx265.so" ]] || die "Missing installed libx265.so in $build/install/lib"
    step "configure_${label}_shim"
    cmake -S "$REPO/prism/codec/native" -B "$shim" -G 'Unix Makefiles' \
        -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release \
        -DX265_INCLUDE_DIR="$build/install/include" \
        -DX265_LIBRARY="$build/install/lib/libx265.so" \
        -DCMAKE_BUILD_RPATH="$build/install/lib"
    step "build_${label}_shim"
    cmake --build "$shim" --parallel "$JOBS"
    [[ -f "$shim/libprism_x265.so" ]] || die "Missing shim in $shim"
}
build_x265 "$GATE/src/x265" "$GATE/build/x265" "$GATE/build/shim" patched
build_x265 "$GATE/src/x265-stock" "$GATE/build/stock/x265" "$GATE/build/stock/shim" stock

step clone_HM_16.2
checkout_tag "$HM_URL" HM-16.2 "$GATE/src/HM"
git -C "$GATE/src/HM" diff --exit-code HEAD
step build_HM_decoder
mkdir -p "$GATE/build/hm/bin"
# HM-16.2 predates modern gcc defaults. Keep warnings visible without -Werror;
# append these flags after HM's own flags and use its original C++98 dialect.
cat > "$GATE/build/hm/cxx" <<'COMPILER'
#!/usr/bin/env bash
exec g++ "$@" -Wno-error -std=gnu++98
COMPILER
chmod +x "$GATE/build/hm/cxx"
for component in TLibCommon TLibVideoIO TLibDecoder TAppCommon; do
    step "build_HM_${component}"
    make -C "$GATE/src/HM/build/linux/lib/$component" -j "$JOBS" \
        CXX="$GATE/build/hm/cxx" release
done
step build_HM_TAppDecoderStatic
make -C "$GATE/src/HM/build/linux/app/TAppDecoder" -j "$JOBS" \
    CXX="$GATE/build/hm/cxx" release
[[ -x "$GATE/src/HM/bin/TAppDecoderStatic" ]] || die 'HM build did not produce bin/TAppDecoderStatic.'
cp -- "$GATE/src/HM/bin/TAppDecoderStatic" "$GATE/build/hm/bin/TAppDecoderStatic"

probe_pin() {
    local library=$1 shim=$2 pin=$3
    # Probe in separate processes: identical SONAMEs must not share a loader.
    PRISM_X265_LIBRARY="$library" PRISM_X265_SHIM="$shim" \
    LD_LIBRARY_PATH="$(dirname -- "$library")" \
    "$PYTHON" - "$pin" <<'PY'
from dataclasses import asdict
import json
from pathlib import Path
import sys
from prism.codec.x265_bridge import X265Library
codec = X265Library()
pin = asdict(codec.pin)
assert pin['abi'] > 0 and pin['picture_size'] > 0
Path(sys.argv[1]).write_text(json.dumps(pin, indent=2) + '\n', encoding='utf-8')
print(json.dumps(pin, indent=2))
PY
}
step pin_patched_build
probe_pin "$GATE/build/x265/install/lib/libx265.so" "$GATE/build/shim/libprism_x265.so" "$GATE/build-pin.json"
step pin_stock_build
probe_pin "$GATE/build/stock/x265/install/lib/libx265.so" "$GATE/build/stock/shim/libprism_x265.so" "$GATE/stock-pin.json"
step record_provenance
"$PYTHON" - "$GATE" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
gate = Path(sys.argv[1])
def commit(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
stack = {
    'instrument': 'x265_prism_residual_v1', 'patch_audit': 'pending first source/build review',
    'repo_commit': commit(gate.parent), 'x265_tag': '3.5',
    'x265_commit': commit(gate / 'src/x265'),
    'stock_commit': commit(gate / 'src/x265-stock'),
    'hm_tag': 'HM-16.2', 'hm_commit': commit(gate / 'src/HM'),
    'patch_sha256': sha(gate / 'x265_residual_patch.patch'),
    'hm_sha256': sha(gate / 'build/hm/bin/TAppDecoderStatic'),
    'hm_compiler_policy': 'g++ -Wno-error -std=gnu++98; decoder and dependencies only',
    'library': str(gate / 'build/x265/install/lib/libx265.so'),
    'shim': str(gate / 'build/shim/libprism_x265.so'),
    'stock_library': str(gate / 'build/stock/x265/install/lib/libx265.so'),
    'stock_shim': str(gate / 'build/stock/shim/libprism_x265.so'),
    'reference_decoder': str(gate / 'build/hm/bin/TAppDecoderStatic'),
}
assert stack['x265_commit'] == stack['stock_commit']
(gate / 'stack.json').write_text(json.dumps(stack, indent=2) + '\n', encoding='utf-8')
PY
step complete
printf 'GATE 0 stack built. Pins identify artifacts; they do not imply a GATE 0 PASS.\n'
