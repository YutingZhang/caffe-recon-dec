#!/bin/bash

SCRIPT_PATH="$0"
SCRIPT_DIR=`dirname "$0"`
CUR_DIR=`pwd`
cd "$SCRIPT_DIR"
SCRIPT_DIR=`pwd -P`
cd "$CUR_DIR"

cd "$SCRIPT_DIR"
find . -name '*_deploy.prototxt' | sort |  while read line; do
    MODEL_SUB_PATH=`echo "$line" | sed -e 's/_deploy\.prototxt$//'`
    cd "$CUR_DIR"
    if [ -e $"SCRIPT_DIR"/"$MODEL_SUB_PATH.caffemodel" ]; then
        echo "[downloaded] ""$MODEL_SUB_PATH"
    else
        echo "             ""$MODEL_SUB_PATH"
    fi
done

