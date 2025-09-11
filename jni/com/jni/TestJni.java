package com.jni;

import java.lang.Math;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public class TestJni {

    private static int SHAPE_0 = 16;
    private static int SHAPE_1 = 128;

    private static InputClass constructInput() {
        InputClass input = new InputClass();
        int uniqueInt = 1;
        input.inputField0D = uniqueInt++;
        input.inputField1D = new int[SHAPE_1];
        for (int i = 0; i < SHAPE_1; i++) {
            input.inputField1D[i] = uniqueInt++;
        }
        input.inputField2D = new int[SHAPE_0][SHAPE_1];
        for (int i = 0; i < SHAPE_0; i++) {
            for (int j = 0; j < SHAPE_1; j++) {
                input.inputField2D[i][j] = uniqueInt++;
            }
        }
        input.inputFieldInner = new InputClassInner();
        input.inputFieldInner.inputFieldInner0D = uniqueInt++;
        return input;
    }

    private static void printOutput(OutputClass output) {
        System.out.println("outputField0D: " + output.outputField0D);
        for (int i = 0; i < Math.min(3, output.outputField1D.length); i++) {
            System.out.println("outputField1D[" + i + "]: " + output.outputField1D[i]);
        }
        for (int i = 0; i < Math.min(3, output.outputField2D.length); i++) {
            for (int j = 0; j < Math.min(3, output.outputField2D[i].length); j++) {
                System.out.println("outputField2D[" + i + "][" + j + "]: " + output.outputField2D[i][j]);
            }
        }
        System.out.println("outputFieldInner.outputFieldInner0D: " + output.outputFieldInner.outputFieldInner0D);
    }

    // Helper to look up native symbols
    public static MemorySegment lookup(String symbol) {
        return Linker.nativeLinker().defaultLookup().find(symbol)
                .or(() -> SymbolLookup.loaderLookup().find(symbol))
                .orElseThrow();
    }

    // MethodHandle for the native gettid() function
    private static final MethodHandle GETTID = Linker.nativeLinker().downcallHandle(
            lookup("gettid"),
            FunctionDescriptor.of(ValueLayout.JAVA_INT));

    // Java wrapper for calling gettid()
    public static int gettid() throws Throwable {
        return (int) GETTID.invokeExact();
    }

    public static void main(String[] args) throws Throwable {

        // Print thread ID info. Will compare with the info printed in C++ side.
        System.out.println("JVM-level Thread ID: " + Thread.currentThread().getId());
        System.out.println("OS-level Thread ID (report by Java): " + gettid());

        InputClass input = constructInput();

        JniMain jni = new JniMain();

        jni.construct();

        OutputClass output = jni.process(input);

        jni.destroy();

        printOutput(output);
    }
}
