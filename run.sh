fp=$1
stem=$(basename $fp)
path=$(dirname $fp)
path=$(realpath $path)
source .env

VLA_NAME=$stem \
    uv run torchrun \
    --standalone \
    run.py \
    --config-path=$path \
    --config-name=$stem \
    ${@:2}
