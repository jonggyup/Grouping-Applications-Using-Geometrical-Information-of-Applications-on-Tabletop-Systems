#!/bin/bash

sed '/CANVAS/!d' $1 > a.c
cat a.c | sed 's/\|/ /'|awk '{print $4, $5, $6}' > output.log

