#include "com_jni_JniMain.h"
#include "util.h"

#include <vector>
#include <iostream>

using namespace std;

const int kNumTrials = 100;

// data structures
struct Input
{
    int inputField0D;
    vector<int> inputField1D;
    vector<vector<int>> inputField2D;
    int inputFieldInner0D;
};

struct Output
{
    int outputField0D;
    vector<int> outputField1D;
    vector<vector<int>> outputField2D;
    int outputFieldInner0D;
};

struct TimerRecord
{
    long timeMicroSecParseInput0D = 0;
    long timeMicroSecParseInput1D = 0;
    long timeMicroSecParseInput2D = 0;
    long timeMicroSecParseInputInner = 0;
    long timeMicroSecUpdateOutput0D = 0;
    long timeMicroSecUpdateOutput1D = 0;
    long timeMicroSecUpdateOutput2D = 0;
    long timeMicroSecUpdateOutputInner = 0;
};

// parse input functions
void parseInput0D(JNIEnv *jenv, jobject jobj_input, Input &input, TimerRecord &timerRecord)
{
    Timer timer;
    timer.tic();
    jclass jcls_input = jenv->GetObjectClass(jobj_input);
    jfieldID jfid_inputField0D = jenv->GetFieldID(jcls_input, "inputField0D", "I");
    input.inputField0D = jenv->GetIntField(jobj_input, jfid_inputField0D);
    timerRecord.timeMicroSecParseInput0D += timer.tocMicroSec();
}

void parseInput1D(JNIEnv *jenv, jobject jobj_input, Input &input, TimerRecord &timerRecord)
{
    Timer timer;
    timer.tic();
    jclass jcls_input = jenv->GetObjectClass(jobj_input);
    jfieldID jfid_inputField1D = jenv->GetFieldID(jcls_input, "inputField1D", "[I");
    jintArray jarr_inputField1D = (jintArray)jenv->GetObjectField(jobj_input, jfid_inputField1D);
    jsize jsize_inputField1D = jenv->GetArrayLength(jarr_inputField1D);
    input.inputField1D.resize(jsize_inputField1D);
    jenv->GetIntArrayRegion(jarr_inputField1D, 0, jsize_inputField1D, input.inputField1D.data());
    jenv->DeleteLocalRef(jarr_inputField1D);
    timerRecord.timeMicroSecParseInput1D += timer.tocMicroSec();
}

void parseInput2D(JNIEnv *jenv, jobject jobj_input, Input &input, TimerRecord &timerRecord)
{
    Timer timer;
    timer.tic();
    jclass jcls_input = jenv->GetObjectClass(jobj_input);
    jfieldID jfid_inputField2D = jenv->GetFieldID(jcls_input, "inputField2D", "[[I");
    jobjectArray jarr_inputField2D = (jobjectArray)jenv->GetObjectField(jobj_input, jfid_inputField2D);
    jsize jsize_inputField2D = jenv->GetArrayLength(jarr_inputField2D);
    input.inputField2D.resize(jsize_inputField2D);
    for (int i = 0; i < jsize_inputField2D; i++)
    {
        jintArray jarr_inputField2D_i = (jintArray)jenv->GetObjectArrayElement(jarr_inputField2D, i);
        jsize jsize_inputField2D_i = jenv->GetArrayLength(jarr_inputField2D_i);
        input.inputField2D[i].resize(jsize_inputField2D_i);
        jenv->GetIntArrayRegion(jarr_inputField2D_i, 0, jsize_inputField2D_i, input.inputField2D[i].data());
        jenv->DeleteLocalRef(jarr_inputField2D_i);
    }
    jenv->DeleteLocalRef(jarr_inputField2D);
    timerRecord.timeMicroSecParseInput2D += timer.tocMicroSec();
}

// reference: https://stackoverflow.com/questions/32513413/jni-getfieldid-returning-null-for-inner-class
void parseInputInner(JNIEnv *env, jobject jobj_input, Input &input, TimerRecord &timerRecord)
{
    Timer timer;
    timer.tic();
    jclass jcls_input = env->GetObjectClass(jobj_input);
    jfieldID jfieldID_innerFieldInner = env->GetFieldID(jcls_input, "inputFieldInner", "Lcom/jni/InputClassInner;");
    jobject jobj_innerFieldInner = env->GetObjectField(jobj_input, jfieldID_innerFieldInner);

    if (jobj_innerFieldInner == nullptr)
    {
        return;
    }

    jclass jcls_innerFieldInner = env->GetObjectClass(jobj_innerFieldInner);
    jfieldID jfieldID_inputFieldInner0D = env->GetFieldID(jcls_innerFieldInner, "inputFieldInner0D", "I");
    input.inputFieldInner0D = env->GetIntField(jobj_innerFieldInner, jfieldID_inputFieldInner0D);

    env->DeleteLocalRef(jobj_innerFieldInner);
    timerRecord.timeMicroSecParseInputInner += timer.tocMicroSec();
}


// update output functions
void updateOutput0D(JNIEnv *jenv, jobject jobj_output, Output &output, TimerRecord &timerRecord)
{
    Timer timer;
    timer.tic();
    jclass jcls_output = jenv->GetObjectClass(jobj_output);
    jfieldID jfid_outputField0D = jenv->GetFieldID(jcls_output, "outputField0D", "I");
    jenv->SetIntField(jobj_output, jfid_outputField0D, output.outputField0D);
    timerRecord.timeMicroSecUpdateOutput0D += timer.tocMicroSec();
}

void updateOutput1D(JNIEnv *jenv, jobject jobj_output, Output &output, TimerRecord &timerRecord)
{
    Timer timer;
    timer.tic();
    jclass jcls_output = jenv->GetObjectClass(jobj_output);
    jfieldID jfid_outputField1D = jenv->GetFieldID(jcls_output, "outputField1D", "[I");
    jintArray jarr_outputField1D = jenv->NewIntArray(output.outputField1D.size());
    jenv->SetIntArrayRegion(jarr_outputField1D, 0, output.outputField1D.size(), output.outputField1D.data());
    jenv->SetObjectField(jobj_output, jfid_outputField1D, jarr_outputField1D);
    jenv->DeleteLocalRef(jarr_outputField1D);
    timerRecord.timeMicroSecUpdateOutput1D += timer.tocMicroSec();
}

void updateOutput2D(JNIEnv *jenv, jobject jobj_output, Output &output, TimerRecord &timerRecord)
{
    Timer timer;
    timer.tic();
    jclass jcls_output = jenv->GetObjectClass(jobj_output);
    jfieldID jfid_outputField2D = jenv->GetFieldID(jcls_output, "outputField2D", "[[I");
    jobjectArray jarr_outputField2D = jenv->NewObjectArray(output.outputField2D.size(), jenv->FindClass("[I"), NULL);
    for (int i = 0; i < output.outputField2D.size(); i++)
    {
        jintArray jarr_outputField2D_i = jenv->NewIntArray(output.outputField2D[i].size());
        jenv->SetIntArrayRegion(jarr_outputField2D_i, 0, output.outputField2D[i].size(), output.outputField2D[i].data());
        jenv->SetObjectArrayElement(jarr_outputField2D, i, jarr_outputField2D_i);
        jenv->DeleteLocalRef(jarr_outputField2D_i);
    }
    jenv->SetObjectField(jobj_output, jfid_outputField2D, jarr_outputField2D);
    jenv->DeleteLocalRef(jarr_outputField2D);
    timerRecord.timeMicroSecUpdateOutput2D += timer.tocMicroSec();
}

void updateOutputInner(JNIEnv *env, jobject jobj_output, Output &output, TimerRecord &timerRecord)
{   
    Timer timer;
    timer.tic();
    jclass jcls_outputClassInner = env->FindClass("com/jni/OutputClassInner");
    jobject jobj_outputClassInner = env->AllocObject(jcls_outputClassInner);

    jfieldID jfieldID_outputFieldInner0D = env->GetFieldID(jcls_outputClassInner, "outputFieldInner0D", "I");
    env->SetIntField(jobj_outputClassInner, jfieldID_outputFieldInner0D, output.outputFieldInner0D);

    jclass jcls_output = env->GetObjectClass(jobj_output);
    jfieldID jfieldID_outputClassInner = env->GetFieldID(jcls_output, "outputFieldInner", "Lcom/jni/OutputClassInner;");
    env->SetObjectField(jobj_output, jfieldID_outputClassInner, jobj_outputClassInner);

    env->DeleteLocalRef(jobj_outputClassInner);
    timerRecord.timeMicroSecUpdateOutputInner += timer.tocMicroSec();
}

// Core class
class Core
{
public:
    Output process(Input input)
    {
        Output output;

        // process outputField0D
        output.outputField0D = input.inputField0D;
        cout << "inputField0D: " << input.inputField0D << endl;

        // process outputField1D
        output.outputField1D = input.inputField1D;
        for (int i = 0; i < min(10UL, input.inputField1D.size()); i++)
        {
            cout << "inputField1D[" << i << "]: " << input.inputField1D[i] << endl;
        }
        
        // process outputField2D
        output.outputField2D = input.inputField2D;
        for (int i = 0; i < min(10UL, input.inputField2D.size()); i++)
        {
            for (int j = 0; j < min(10UL, input.inputField2D[i].size()); j++)
            {
                cout << "inputField2D[" << i << "][" << j << "]: " << input.inputField2D[i][j] << endl;
            }
        }

        // process outputFieldInner0D
        output.outputFieldInner0D = input.inputFieldInner0D;
        cout << "inputFieldInner0D: " << input.inputFieldInner0D << endl;

        return output;
    }
};

// JNI functions
JNIEXPORT jlong JNICALL Java_com_jni_JniMain_c_1constructCore
  (JNIEnv *, jobject)
{
    return (jlong)(new Core());
}

JNIEXPORT jobject JNICALL Java_com_jni_JniMain_c_1process
  (JNIEnv * jenv, jobject, jlong jlong_corePtr, jobject jobj_input)
{
    Core &core = *((Core*)jlong_corePtr);
    TimerRecord timerRecord;

    Input input;
    for (int i = 0; i < kNumTrials; i++)
    {
        parseInput0D(jenv, jobj_input, input, timerRecord);
        parseInput1D(jenv, jobj_input, input, timerRecord);
        parseInput2D(jenv, jobj_input, input, timerRecord);
        parseInputInner(jenv, jobj_input, input, timerRecord);
    }

    Output output = core.process(input);

    jclass jcls_output = jenv->FindClass("com/jni/OutputClass");
    jobject jobj_output = jenv->AllocObject(jcls_output);

    for (int i = 0; i < kNumTrials; i++)
    {
        updateOutput0D(jenv, jobj_output, output, timerRecord);
        updateOutput1D(jenv, jobj_output, output, timerRecord);
        updateOutput2D(jenv, jobj_output, output, timerRecord);
        updateOutputInner(jenv, jobj_output, output, timerRecord);
    }

    cout << "timeMicroSecParseInput0D: " << (double)timerRecord.timeMicroSecParseInput0D / kNumTrials << endl;
    cout << "timeMicroSecParseInput1D: " << (double)timerRecord.timeMicroSecParseInput1D / kNumTrials << endl;
    cout << "timeMicroSecParseInput2D: " << (double)timerRecord.timeMicroSecParseInput2D / kNumTrials << endl;
    cout << "timeMicroSecParseInputInner: " << (double)timerRecord.timeMicroSecParseInputInner / kNumTrials << endl;
    cout << "timeMicroSecUpdateOutput0D: " << (double)timerRecord.timeMicroSecUpdateOutput0D / kNumTrials << endl;
    cout << "timeMicroSecUpdateOutput1D: " << (double)timerRecord.timeMicroSecUpdateOutput1D / kNumTrials << endl;
    cout << "timeMicroSecUpdateOutput2D: " << (double)timerRecord.timeMicroSecUpdateOutput2D / kNumTrials << endl;
    cout << "timeMicroSecUpdateOutputInner: " << (double)timerRecord.timeMicroSecUpdateOutputInner / kNumTrials << endl;

    return jobj_output;
}

JNIEXPORT void JNICALL Java_com_jni_JniMain_c_1destroyCore
  (JNIEnv *, jobject, jlong jCore)
{
    delete (Core*)jCore;
}
