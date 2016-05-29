#!/bin/bash

SCRIPT_PATH="$0"
SCRIPT_DIR=`dirname "$0"`
CUR_DIR=`pwd`
cd "$SCRIPT_DIR"
SCRIPT_DIR=`pwd -P`
cd "$CUR_DIR"

MODEL_PATH=$1

if [ "$MODEL_PATH" == "" ]; then
    cd "$SCRIPT_DIR"
    find . -name '*_deploy.prototxt' | sort |  while read line; do
        MODEL_SUB_PATH=`echo "$line" | sed -e 's/_deploy\.prototxt$//'`
        cd "$CUR_DIR"
        "$0" "$MODEL_SUB_PATH"
    done
    exit
fi

URL_PREFIX="`cat $SCRIPT_DIR/url_prefix.txt`"

SUB_FOLDER=`dirname "$MODEL_PATH"`
cd $SCRIPT_DIR
if [ "$SUB_FOLDER" != "" ]; then
    mkdir -p $SUB_FOLDER
    cd "$SUB_FOLDER"
fi

wget -c "$URL_PREFIX""$MODEL_PATH".caffemodel


