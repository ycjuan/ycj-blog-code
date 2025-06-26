#include "com_jni_JniMain.h"
#include "core.h"
#include "util.h"

#include <vector>
#include <iostream>

using namespace std;

JNIEXPORT jlong JNICALL Java_com_jni_JniMain_c_1constructCore
  (JNIEnv *, jobject)
{
    return (jlong)(new Core());
}
JNIEXPORT void JNICALL Java_com_jni_JniMain_c_1process
  (JNIEnv * jenv, jobject, jlong jlong_corePtr, int numThreads)
{
    Core &core = *((Core*)jlong_corePtr);
    core.process(numThreads);
}

JNIEXPORT void JNICALL Java_com_jni_JniMain_c_1destroyCore
  (JNIEnv *, jobject, jlong jong_corePtr)
{
    delete (Core*)jong_corePtr;
}