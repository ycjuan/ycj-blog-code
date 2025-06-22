#!/bin/bash

rm -f dump1
jmap -dump:format=b,file=dump1 $1
