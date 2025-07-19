The purpose of this experiment code is trying to answer the question:

    How much more expensive is it to run multi-layer perception (MLP) compared to a simple dot-product (DP)

This experiment code include the following methods:

  - method_dp_cpu.cuh: an implementation of CPU verison of DP.

  - method_dp_gpu_naive.cuh: an naive implementation of DP with a simple Cuda kernel

  - method_dp_gpu_cublas.cuh: an implementation of DP using Cublas

  - method_mlp_cpu.cuh: an implementation of CPU verison of MLP.

  - method_mlp_gpu.cuh: an place holder for the "challenger" to implement MLP in GPU

Since this is just an experimental code, we put everything under data.cuh:

  - numDocs: number of documents
  
  - numReqs: number of requests (aka batch size)

  - embDim: dimension of req / doc embeddings

  - hiddenDim: only used in MLP. This indicates the dimension of hidden layer

  - d_doc: used to store the `numDocs * embDim` matrix

  - d_req: used to store the `numReqs * embDim` matrix

  - d_wa: used to store the `embDim * hiddenDim` matrix used by MLP

  - d_wb: used to store the `hiddenDim * 1` matrix used by MLP

  - xxxMemLayout: used to indicate row-major or col-major of each matrix



To the challenger
=================

So the challenger is to implement `method_mlp_gpu`, and compare its speed with `method_dp_gpu_cublas`. Here is what I think the challenges are:

1. I think the Hadamard Product step is going to be a bottleneck since I don't think we can use tensor cores to compute it.

2. When you write the code, you may be tempted to store things in a temporary `numDocs * numReqs * embDim` matrix. This is forbiddened because it is too big.

3. Simply from computational complexity analysis, from DP to MLP, it grows from `O(numDocs * numReq * embDim)` to `O(numDocs * numReq * embDim * hiddenDim)`. So MLP is `hiddenDim` times more expensive then DP in theoretical analysis.

Also, before you start working on GPU version, please take a look at `method_mlp_cpu.cuh` to make sure the implementaion is correct.