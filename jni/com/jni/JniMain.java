package com.jni;

public class JniMain {

    static {
        System.loadLibrary("core");
    }

    public void construct() {
        core_ptr = c_constructCore();
    }

    public OutputClass process(InputClass input) {

        OutputClass output = c_process(core_ptr, input);

        return output;
    }

    public void destroy() {
        c_destroyCore(core_ptr);
    }

    private native long c_constructCore();

    private native OutputClass c_process(long eng_ptr, InputClass input);

    private native void c_destroyCore(long eng_ptr);

    private long core_ptr;
}
