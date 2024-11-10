#include <jni.h>
#include "com_retrieval_RetrievalEngineJNI.h"  // Generated
#include <iostream>
#include <vector>
#include <cassert>

#ifndef DATA_STRUCTS_H
#define DATA_STRUCTS_H
#include "../data_structs.h"
#endif

#ifndef RETRIEVAL_ENGINE_H
#define RETRIEVAL_ENGINE_H
#include "../retrieval_engine.h"
#endif

#ifndef COMMON_H
#define COMMON_H
#include "../common.h"
#endif

#ifndef CPU_OPS_H
#define CPU_OPS_H
#include "../cpu_ops.h"
#endif


namespace {
    using namespace std;
}

JNIEXPORT jlong JNICALL Java_com_retrieval_RetrievalEngineJNI_c_1constructRetrievalEngine
  (JNIEnv *j_env, jobject, jobjectArray j_tbr3D, jobjectArray j_ebr2D, jfloatArray j_bid1D, jlongArray j_rowIdToItemIdMap, jint j_numDocs, jint j_ebrDim, jint j_numClauses) {
    
    printf("Constructing Cuda Retrieval Engine");
    Timer timer;
    int numDocs    = (int)j_numDocs;
    int ebrDim     = (int)j_ebrDim;
    int numClauses = (int)j_numClauses;

    // Copy EBR data from Java to C
    printf("Copy EBR data from Java to C...\n");
    vector<vector<float>> ebr2D(numDocs);

    for (int i = 0; i < numDocs; i++) {
        jfloatArray j_emb1D = (jfloatArray) j_env->GetObjectArrayElement(j_ebr2D, i);
        ebr2D[i].resize(ebrDim);
        j_env->GetFloatArrayRegion(j_emb1D, 0L, (long)ebrDim, ebr2D[i].data());
        j_env->DeleteLocalRef(j_emb1D);
    }
    timer.toc();
    printf("Copy EBR data from Java to C...done (%f ms)\n", timer.getms());
    
    // Copy TBR data from Java to C
    printf("Copy TBR data from Java to C...\n");
    vector<vector<vector<long>>> tbr3D(numDocs);
    for (int i = 0; i < numDocs; i++) {
        jobjectArray j_tbr2D = (jobjectArray) j_env->GetObjectArrayElement(j_tbr3D, i);
        tbr3D[i].resize(numClauses);
        for (int c = 0; c < numClauses; c++) {
            jlongArray j_hashes = (jlongArray) j_env->GetObjectArrayElement(j_tbr2D, c);
            int numAttrs = (int)j_env->GetArrayLength(j_hashes);
            tbr3D[i][c].resize(numAttrs);
            j_env->GetLongArrayRegion(j_hashes, 0L, (long)numAttrs, tbr3D[i][c].data());
            j_env->DeleteLocalRef(j_hashes);
        }
        j_env->DeleteLocalRef(j_tbr2D);
    }
    timer.toc();
    printf("Copy TBR data from Java to C...done (%f ms)\n", timer.getms());

    // Copy bid1D from Java to C
    printf("Copy bid data from Java to C...\n");
    vector<float> bid1D(numDocs);
    j_env->GetFloatArrayRegion(j_bid1D, 0L, (long)numDocs, bid1D.data());
    printf("Copy bid data from Java to C...done\n");

    // Copy rowIdToItemIdMap from Java to C
    printf("Copy rowIdToItemIdMap from Java to C...\n");
    vector<long> rowIdToItemIdMap(numDocs);
    j_env->GetLongArrayRegion(j_rowIdToItemIdMap, 0L, (long)numDocs, rowIdToItemIdMap.data());
    printf("Copy rowIdToItemIdMap from Java to C...done\n");

    RetrievalEngine *cs = new RetrievalEngine(tbr3D, ebr2D, bid1D, rowIdToItemIdMap);

    return (jlong)cs;
}

namespace {

    vector<int> retrieveIntVector(JNIEnv *j_env, jintArray j_intArray) {
        int length = (int)j_env->GetArrayLength(j_intArray);
        vector<int> intVec(length);
        j_env->GetIntArrayRegion(j_intArray, 0L, (long)length, intVec.data());
        return intVec;
    }

    vector<float> retrieveEmb(JNIEnv *j_env, jfloatArray j_emb1D) {
        int ebrDim = (int)j_env->GetArrayLength(j_emb1D);
        vector<float> emb1D(ebrDim);
        j_env->GetFloatArrayRegion(j_emb1D, 0L, (long)ebrDim, emb1D.data());
        return emb1D;
    }

    vector<vector<long>> retrieveHashes(JNIEnv *j_env, jobjectArray j_tbr2D) {
        int numClauses = (int)j_env->GetArrayLength(j_tbr2D);
        vector<vector<long>> tbr2D;
        for (int c = 0; c < numClauses; c++) {
            jlongArray j_hashes = (jlongArray) j_env->GetObjectArrayElement(j_tbr2D, c);
            int numHashes = (int)j_env->GetArrayLength(j_hashes);
            vector<long> hashes(numHashes);
            j_env->GetLongArrayRegion(j_hashes, 0L, (long)numHashes, hashes.data());
            tbr2D.push_back(hashes);
            j_env->DeleteLocalRef(j_hashes);
        }
        return tbr2D;
    }
}

JNIEXPORT jint JNICALL Java_com_retrieval_RetrievalEngineJNI_c_1retrieve
    (JNIEnv *j_env, jobject, jlong j_cs_ptr, jobjectArray j_tbr2D, jfloatArray j_emb1D, jintArray j_tbrOps, jint j_k, jint j_quantK, jlongArray j_itemIds, jfloatArray j_scores, jfloatArray j_cudaTimeMs) {

    int k = (int)j_k;
    int quantK = (int)j_quantK;
    RetrievalEngine &cs = *(RetrievalEngine*)j_cs_ptr;

    vector<int> tbrOp1D = retrieveIntVector(j_env, j_tbrOps);
    vector<float> emb1D = retrieveEmb(j_env, j_emb1D);
    vector<vector<long>> tbr2D = retrieveHashes(j_env, j_tbr2D);

    Timer timer;
    vector<vector<float>> ebr2D = { emb1D };
    vector<vector<vector<long>>> tbr3D = { tbr2D };
    RetrievedResults results = cs.retrieveGpu(tbr3D, ebr2D, tbrOp1D, k, quantK);
    timer.toc();
    float cudaTimeMs = timer.getms();
    j_env->SetFloatArrayRegion(j_cudaTimeMs, 0L, 1L, &cudaTimeMs);
     
    vector<Doc> doc1D = results.doc2D[0];
    vector<long> docId1D(doc1D.size());
    vector<float> score1D(doc1D.size());
    j_env->SetLongArrayRegion(j_itemIds, 0L, (long)docId1D.size(), docId1D.data());
    j_env->SetFloatArrayRegion(j_scores, 0L, (long)score1D.size(), score1D.data());

    return (jint)doc1D.size();
}

JNIEXPORT void JNICALL Java_com_retrieval_RetrievalEngineJNI_c_1destroy
  (JNIEnv *, jobject, jlong j_cs_ptr) {

    RetrievalEngine *cs = (RetrievalEngine*)j_cs_ptr;
    delete cs;
}
