#include "com_jni_JniMain.h"

#include <vector>
#include <iostream>

using namespace std;

// data structures
struct Input
{
    int inputField0D;
    vector<int> inputField1D;
    vector<vector<int>> inputField2D;
};

struct Output
{
    int outputField0D;
    vector<int> outputField1D;
    vector<vector<int>> outputField2D;
};

// parse input functions
void parseInput0D(JNIEnv *jenv, jobject jobj_input, Input &input)
{
    jclass jcls_input = jenv->GetObjectClass(jobj_input);
    jfieldID jfid_inputField0D = jenv->GetFieldID(jcls_input, "inputField0D", "I");
    input.inputField0D = jenv->GetIntField(jobj_input, jfid_inputField0D);
}

void parseInput1D(JNIEnv *jenv, jobject jobj_input, Input &input)
{
    jclass jcls_input = jenv->GetObjectClass(jobj_input);
    jfieldID jfid_inputField1D = jenv->GetFieldID(jcls_input, "inputField1D", "[I");
    jintArray jarr_inputField1D = (jintArray)jenv->GetObjectField(jobj_input, jfid_inputField1D);
    jsize jsize_inputField1D = jenv->GetArrayLength(jarr_inputField1D);
    input.inputField1D.resize(jsize_inputField1D);
    jenv->GetIntArrayRegion(jarr_inputField1D, 0, jsize_inputField1D, input.inputField1D.data());
    jenv->DeleteLocalRef(jarr_inputField1D);
}

void parseInput2D(JNIEnv *jenv, jobject jobj_input, Input &input)
{
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
}

// write output functions
void writeOutput0D(JNIEnv *jenv, jobject jobj_output, Output &output)
{
    jclass jcls_output = jenv->GetObjectClass(jobj_output);
    jfieldID jfid_outputField0D = jenv->GetFieldID(jcls_output, "outputField0D", "I");
    jenv->SetIntField(jobj_output, jfid_outputField0D, output.outputField0D);
}

void writeOutput1D(JNIEnv *jenv, jobject jobj_output, Output &output)
{
    jclass jcls_output = jenv->GetObjectClass(jobj_output);
    jfieldID jfid_outputField1D = jenv->GetFieldID(jcls_output, "outputField1D", "[I");
    jintArray jarr_outputField1D = jenv->NewIntArray(output.outputField1D.size());
    jenv->SetIntArrayRegion(jarr_outputField1D, 0, output.outputField1D.size(), output.outputField1D.data());
    jenv->SetObjectField(jobj_output, jfid_outputField1D, jarr_outputField1D);
    jenv->DeleteLocalRef(jarr_outputField1D);
}

void writeOutput2D(JNIEnv *jenv, jobject jobj_output, Output &output)
{
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
}

// Core class
class Core
{
public:
    Output process(Input input)
    {
        Output output;
        output.outputField0D = input.inputField0D;
        cout << "inputField0D: " << input.inputField0D << endl;
        output.outputField1D = input.inputField1D;
        for (int i = 0; i < input.inputField1D.size(); i++)
        {
            cout << "inputField1D[" << i << "]: " << input.inputField1D[i] << endl;
        }
        output.outputField2D = input.inputField2D;
        for (int i = 0; i < input.inputField2D.size(); i++)
        {
            for (int j = 0; j < input.inputField2D[i].size(); j++)
            {
                cout << "inputField2D[" << i << "][" << j << "]: " << input.inputField2D[i][j] << endl;
            }
        }
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

    Input input;
    parseInput0D(jenv, jobj_input, input);
    parseInput1D(jenv, jobj_input, input);
    parseInput2D(jenv, jobj_input, input);

    Output output = core.process(input);

    jclass jcls_output = jenv->FindClass("com/jni/OutputClass");
    jobject jobj_output = jenv->AllocObject(jcls_output);
    writeOutput0D(jenv, jobj_output, output);
    writeOutput1D(jenv, jobj_output, output);
    writeOutput2D(jenv, jobj_output, output);

    return jobj_output;
}

JNIEXPORT void JNICALL Java_com_jni_JniMain_c_1destroyCore
  (JNIEnv *, jobject, jlong jCore)
{
    delete (Core*)jCore;
}
