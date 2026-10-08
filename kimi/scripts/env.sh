# Sourced by the other scripts: makes kimi/ and the Kimi-Audio submodule importable.
KIMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIMI_AUDIO_DIR="$KIMI_DIR/third_party/Kimi-Audio"
export PYTHONPATH="$KIMI_DIR:$KIMI_AUDIO_DIR:${PYTHONPATH:-}"
