package com.jni;

import java.lang.Math;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class TestJni {

    public static void worker()
    {
        for (int i = 0; i < 100000000L; i++)
        {
            int[] largeArray = new int[10000000];
            for (int j = 0; j < largeArray.length; j++) {
                largeArray[j] = j;
            }
            /*
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
                 */
            //System.gc();
        }
    }
    
    public static void main(String[] args) {

        JniMain jni = new JniMain();

        jni.construct();

        ExecutorService executorService = Executors.newFixedThreadPool(1);

        /*
        Future<Void> future = executorService.submit(() -> {
            while (true) {
                jni.process();
                Thread.sleep(10);
            }
        });
         */

        Future<Void> future = executorService.submit(() -> {
            worker();
            return null;
        });

        for (int i = 0; i < 100000000L; i++)
        {
            jni.process();
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        try {
            future.get();
        } catch (Exception e) {
            e.printStackTrace();
        }

        jni.destroy();

        executorService.shutdown();
    }
}
