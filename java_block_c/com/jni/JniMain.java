package com.jni;

public class JniMain {

    static {
        System.loadLibrary("core");
    }
    public void construct() {
        core_ptr = c_constructCore();
    }
    public void process() {

        c_process(core_ptr);

    }
    public void destroy() {
        c_destroyCore(core_ptr);
    }

    private native long c_constructCore();
    private native void c_process(long core_ptr);
    private native void c_destroyCore(long core_ptr);

    private long core_ptr;
}
