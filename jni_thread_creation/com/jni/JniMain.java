package com.jni;

public class JniMain {

    static {
        System.loadLibrary("core");
    }
    public void construct() {
        core_ptr = c_constructCore();
    }
    public void process(int numThreads) {

        c_process(core_ptr, numThreads);

    }
    public void destroy() {
        c_destroyCore(core_ptr);
    }

    private native long c_constructCore();
    private native void c_process(long core_ptr, int numThreads);
    private native void c_destroyCore(long core_ptr);

    private long core_ptr;
}
