#include "com_jni_JniMain.h"
#include "util.h"

#include <vector>
#include <iostream>

using namespace std;

// Core class
class Core
{
public:
    void process()
    {
        while(true)
        {
            Timer timer;
            timer.tic();
        
            vector<int> vec(2000000);
            for (int i = 0; i < vec.size(); i++)
            {
                vec[i] = i;
            }


            
            vec.clear();
            vec.shrink_to_fit();

            float timeMs = timer.tocMicroSec() / 1000.0;
            if (timeMs > 1)
            {
                cout << "[!!!] " << timeMs << " ms" << endl;
            }
        }
    }
};

JNIEXPORT jlong JNICALL Java_com_jni_JniMain_c_1constructCore
  (JNIEnv *, jobject)
{
    return (jlong)(new Core());
}
JNIEXPORT void JNICALL Java_com_jni_JniMain_c_1process
  (JNIEnv * jenv, jobject, jlong jlong_corePtr)
{
    Core &core = *((Core*)jlong_corePtr);
    core.process();
}

JNIEXPORT void JNICALL Java_com_jni_JniMain_c_1destroyCore
  (JNIEnv *, jobject, jlong jong_corePtr)
{
    delete (Core*)jong_corePtr;
}