#!/bin/bash

set -e
set -x

# REF
# https://www3.ntu.edu.sg/home/ehchua/programming/java/javanativeinterface.html
# https://www.baeldung.com/jni

export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64" # Replace this by the correct path on your server
javac -h . com/jni/JniMain.java # It will generate a header file called com_jni_JniMain.h
g++ -fPIC -std=c++14 -O3 -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux" -shared -o libcore.so jni.cc
javac com/jni/JniMain.java
javac com/jni/TestJni.java
java -Xms1G -Xmx2G -Djava.library.path=. -cp . com.jni.TestJni

g++ -fPIC -std=c++14 -O3 main.cc
./a.out