package com.jni;

import java.lang.Math;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class TestJni {

    public static void main(String[] args) {

        int numTrials = 10000;
        int numThreads = 16;

        JniMain jni = new JniMain();

        jni.construct();

        // Run benchmark
        System.out.println("Running benchmark with " + numTrials + " trials and " + numThreads + " threads");
        long startTime = 0;
        for (int i = -3; i < numTrials; i++)
        {
            if (i == 0) {
                startTime = System.currentTimeMillis();
            }
            jni.process(numThreads);
        }
        long endTime = System.currentTimeMillis();
        long timeSpent = endTime - startTime;
        double avgTime = (double)timeSpent / numTrials;
        System.out.println("Average time per trial: " + avgTime + " ms");

        jni.destroy();

    }
}
