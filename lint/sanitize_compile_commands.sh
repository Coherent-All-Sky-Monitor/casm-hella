#!/usr/bin/bash

# Sanitize compile_commands.json for use with clang-tidy (and clangd)

SOURCE_PATH=$1
COMPILE_COMMANDS=$2

if ! test -f $COMPILE_COMMANDS; then
    echo "Compilation database ($COMPILE_COMMANDS) not found, aborting"
    exit 1
fi

# Scrub directories we don't want to lint
jq '[ . - map(select(.file | contains("/resources/"))) | .[] ]' $COMPILE_COMMANDS | sponge $COMPILE_COMMANDS

# Sanitize CUDA compilation flags
jq '[ .[] | .command |= sub("nvcc";"nvcc -I'"${SOURCE_PATH}"'/lint/patched_headers/ -Xclang -fcuda-allow-variadic-functions --no-cuda-version-check")  ]' $COMPILE_COMMANDS | sponge $COMPILE_COMMANDS
jq '[ .[] | .command |= sub("--use_fast_math";"")  ]' $COMPILE_COMMANDS | sponge $COMPILE_COMMANDS
jq '[ .[] | .command |= sub("-Xcompiler=-fPIC";"")  ]' $COMPILE_COMMANDS | sponge $COMPILE_COMMANDS
jq '[ .[] | .command |= sub("-forward-unknown-to-host-compiler";"")  ]' $COMPILE_COMMANDS | sponge $COMPILE_COMMANDS
jq '[ .[] | .command |= gsub("--generate-code=.*?\\s";"")  ]' $COMPILE_COMMANDS | sponge $COMPILE_COMMANDS
jq '[ .[] | .command |= gsub("--options-file ";"@")  ]' $COMPILE_COMMANDS | sponge $COMPILE_COMMANDS

# Hack for GCC 12 compatibility - we need to enable the patched shared_ptr_base.h if clang is pulling from GCC 12 headers
clang_gcc_inst=`basename $(dirname $(clang -print-libgcc-file-name))` # Output from clang will be something like /usr/bin/../lib/gcc/x86_64-linux-gnu/12/libgcc.a
if [ $clang_gcc_inst -eq 12 ]; then
    jq '[ .[] | .command |= sub("nvcc";"nvcc -I'"${SOURCE_PATH}"'/lint/patched_headers/gcc_12")  ]' $COMPILE_COMMANDS | sponge $COMPILE_COMMANDS
fi
