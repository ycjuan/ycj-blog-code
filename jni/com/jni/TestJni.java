package com.jni;

public class TestJni {

    private static InputClass constructInput() {
        InputClass input = new InputClass();
        input.inputField0D = 1;
        input.inputField1D = new int[] {11, 12, 13};
        input.inputField2D = new int[][] {{21, 22, 23}, {24, 25, 26}};
        return input;
    }

    public static void main(String[] args) {

        InputClass input = constructInput();

        JniMain jni = new JniMain();

        jni.construct();

        OutputClass output = jni.process(input);

        jni.destroy();

        System.out.println("outputField0D: " + output.outputField0D);
        for (int i = 0; i < output.outputField1D.length; i++) {
            System.out.println("outputField1D[" + i + "]: " + output.outputField1D[i]);
        }
        for (int i = 0; i < output.outputField2D.length; i++) {
            for (int j = 0; j < output.outputField2D[i].length; j++) {
                System.out.println("outputField2D[" + i + "][" + j + "]: " + output.outputField2D[i][j]);
            }
        }
    }
}
