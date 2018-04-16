// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#ifndef PATCHMATCH_GPU_H_
#define PATCHMATCH_GPU_H_

#include <cfloat>

#include <curand.h>
#include <curand_kernel.h>

#include "texarray2.h"
#include "memarray2.h"

typedef Vec<1,float> V1f;
typedef Array2<Vec<1,float>> A2V1f;

__global__ void krnlInitRngStates(const int width,
                                  const int height,
                                  curandState* rngStates)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<width && y<height)
  {
    const int idx = x+y*width;
    curand_init((1337 << 20) + idx, 0, 0, &rngStates[idx]);
  }
}

curandState* initGpuRng(const int width,
                        const int height)
{
  curandState* gpuRngStates;
  cudaMalloc(&gpuRngStates,width*height*sizeof(curandState));

  const dim3 threadsPerBlock(16,16);
  const dim3 numBlocks((width+threadsPerBlock.x)/threadsPerBlock.x,
                       (height+threadsPerBlock.y)/threadsPerBlock.y);

  krnlInitRngStates<<<numBlocks,threadsPerBlock>>>(width,height,gpuRngStates);

  return gpuRngStates;
}

template<int N,typename T,int M>
struct PatchSSD
{
  const TexArray2<N,T,M> A;
  const TexArray2<N,T,M> B;
  const Vec<N,float> weights;

  PatchSSD(const TexArray2<N,T,M>& A,
           const TexArray2<N,T,M>& B,
           const Vec<N,float>& weights)

  : A(A),B(B),weights(weights) {}

   __device__ float operator()(int patchWidth,
                               const int ax,
                               const int ay,
                               const int bx,
                               const int by,
                               const float ebest)
   {
    const int hpw = patchWidth/2;
    float ssd = 0;

    for(int py=-hpw;py<=+hpw;py++)
    {
      for(int px=-hpw;px<=+hpw;px++)
      {
        const Vec<N,T> pixelA = A(ax + px, ay + py);
        const Vec<N,T> pixelB = B(bx + px, by + py);
        for(int i=0;i<N;i++)
        {
          const float diff = float(pixelA[i])-float(pixelB[i]);
          ssd += weights[i]*diff*diff;
        }
      }

      if (ssd>ebest) { return ssd; }
    }

    return ssd;
   }
};

template<typename FUNC>
__global__ void krnlEvalErrorPass(const int patchWidth,
                                  FUNC patchError,
                                  const TexArray2<2,int> NNF,
                                  TexArray2<1,float> E)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<NNF.width && y<NNF.height)
  {
    const V2i n = NNF(x,y);
    E.write(x,y,V1f(patchError(patchWidth,x,y,n[0],n[1],FLT_MAX)));
  }
}

void __device__ updateOmega(MemArray2<int>& Omega,const int patchWidth,const int bx,const int by,const int incdec)
{
  const int r = patchWidth/2;

  for(int oy=-r;oy<=+r;oy++)
  for(int ox=-r;ox<=+r;ox++)
  {
    const int x = bx+ox;
    const int y = by+oy;
    atomicAdd(&Omega.data[x+y*Omega.width],incdec);
    //Omega.data[x+y*Omega.width] += incdec;
  }
}

int __device__ patchOmega(const int patchWidth,const int bx,const int by,const MemArray2<int>& Omega)
{
  const int r = patchWidth/2;

  int sum = 0;

  for(int oy=-r;oy<=+r;oy++)
  for(int ox=-r;ox<=+r;ox++)
  {
    const int x = bx+ox;
    const int y = by+oy;
    sum += Omega.data[x+y*Omega.width]; /// XXX: atomic read instead ??
  }

  return sum;
}

template<typename FUNC>
__device__ void tryPatch(const  V2i& sizeA,
                         const  V2i& sizeB,
                                MemArray2<int>& Omega,
                         const  int patchWidth,
                         FUNC   patchError,
                         const  float lambda,
                         const  int ax,
                         const  int ay,
                         const  int bx,
                         const  int by,
                         V2i&   nbest,
                         float& ebest)
{
  const float omegaBest = (float(sizeA(0)*sizeA(1)) /
                           float(sizeB(0)*sizeB(1))) * float(patchWidth*patchWidth);

  const float curOcc = (float(patchOmega(patchWidth,nbest(0),nbest(1),Omega))/float(patchWidth*patchWidth))/omegaBest;
  const float newOcc = (float(patchOmega(patchWidth,      bx,      by,Omega))/float(patchWidth*patchWidth))/omegaBest;

  const float curErr = ebest;
  const float newErr = patchError(patchWidth,ax,ay,bx,by,curErr+lambda*curOcc);

  if ((newErr+lambda*newOcc) < (curErr+lambda*curOcc))
  {
    updateOmega(Omega,patchWidth,      bx,      by,+1);
    updateOmega(Omega,patchWidth,nbest(0),nbest(1),-1);
    nbest = V2i(bx,by);
    ebest = newErr;
  }
}

template<typename FUNC>
__device__ void tryNeighborsOffset(const int x,
                                   const int y,
                                   const int ox,
                                   const int oy,
                                   V2i& nbest,
                                   float& ebest,
                                   const V2i& sizeA,
                                   const V2i& sizeB,
                                         MemArray2<int>& Omega,
                                   const int patchWidth,
                                   FUNC patchError,
                                   const float lambda,
                                   const TexArray2<2,int>& NNF)
{
  const int hpw = patchWidth/2;

  const V2i on = NNF(x+ox,y+oy);
  const int nx = on(0)-ox;
  const int ny = on(1)-oy;

  if (nx>=hpw && nx<sizeB(0)-hpw &&
      ny>=hpw && ny<sizeB(1)-hpw)
  {
    tryPatch(sizeA,sizeB,Omega,patchWidth,patchError,lambda,x,y,nx,ny,nbest,ebest);
  }
}

template<typename FUNC>
__global__ void krnlPropagationPass(const V2i sizeA,
                                    const V2i sizeB,
                                          MemArray2<int> Omega,
                                    const int patchWidth,
                                    FUNC  patchError,
                                    const float lambda,
                                    const int r,
                                    const TexArray2<2,int> NNF,
                                    TexArray2<2,int> NNF2,
                                    TexArray2<1,float> E,
                                    TexArray2<1,unsigned char> mask)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<sizeA(0) && y<sizeA(1))
  {
    V2i   nbest = NNF(x,y);
    float ebest = E(x,y)(0);

    if (mask(x,y)[0]==255)
    {
      tryNeighborsOffset(x,y,-r,0,nbest,ebest,sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF);
      tryNeighborsOffset(x,y,+r,0,nbest,ebest,sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF);
      tryNeighborsOffset(x,y,0,-r,nbest,ebest,sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF);
      tryNeighborsOffset(x,y,0,+r,nbest,ebest,sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF);
    }

    E.write(x,y,V1f(ebest));
    NNF2.write(x,y,nbest);
  }
}

template<typename FUNC>
__device__ void tryRandomOffsetInRadius(const int r,
                                        const V2i& sizeA,
                                        const V2i& sizeB,
                                              MemArray2<int>& Omega,
                                        const int patchWidth,
                                        FUNC  patchError,
                                        const float lambda,
                                        const int x,
                                        const int y,
                                        const V2i& norg,
                                        V2i&  nbest,
                                        float& ebest,
                                        curandState* rngState)
{
  const int hpw = patchWidth/2;

  const int xmin = max(norg(0)-r,hpw);
  const int xmax = min(norg(0)+r,sizeB(0)-1-hpw);
  const int ymin = max(norg(1)-r,hpw);
  const int ymax = min(norg(1)+r,sizeB(1)-1-hpw);

  const int nx = xmin+(curand(rngState)%(xmax-xmin+1));
  const int ny = ymin+(curand(rngState)%(ymax-ymin+1));

  tryPatch(sizeA,sizeB,Omega,patchWidth,patchError,lambda,x,y,nx,ny,nbest,ebest);
}

template<typename FUNC>
__global__ void krnlRandomSearchPass(const V2i sizeA,
                                     const V2i sizeB,
                                     MemArray2<int> Omega,
                                     const int patchWidth,
                                     FUNC  patchError,
                                     const float lambda,
                                     TexArray2<2,int> NNF,
                                     TexArray2<1,float> E,
                                     TexArray2<1,unsigned char> mask,
                                     curandState* rngStates)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<sizeA(0) && y<sizeA(1))
  {
    if (mask(x,y)[0]==255)
    {
      V2i nbest = NNF(x,y);
      float ebest = E(x,y)(0);

      const V2i norg = nbest;

      for(int r=1;r<max(sizeB(0),sizeB(1))/2;r=r*2)
      {
        tryRandomOffsetInRadius(r,sizeA,sizeB,Omega,patchWidth,patchError,lambda,x,y,norg,nbest,ebest,&rngStates[x+y*NNF.width]);
      }

      E.write(x,y,V1f(ebest));
      NNF.write(x,y,nbest);
    }
  }
}

template<typename FUNC>
void patchmatchGPU(const V2i sizeA,
                   const V2i sizeB,
                   MemArray2<int>& Omega,
                   const int patchWidth,
                   FUNC patchError,
                   const float lambda,
                   const int numIters,
                   const int numThreadsPerBlock,
                   TexArray2<2,int>& NNF,
                   TexArray2<2,int>& NNF2,
                   TexArray2<1,float>& E,
                   TexArray2<1,unsigned char>& mask,
                   curandState* rngStates)
{
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
  const dim3 numBlocks = dim3((NNF.width+threadsPerBlock.x)/threadsPerBlock.x,
                              (NNF.height+threadsPerBlock.y)/threadsPerBlock.y);

  krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchWidth,patchError,NNF,E);

  checkCudaError(cudaDeviceSynchronize());

  for(int i=0;i<numIters;i++)
  {
    krnlPropagationPass<<<numBlocks,threadsPerBlock>>>(sizeA,sizeB,Omega,patchWidth,patchError,lambda,4,NNF,NNF2,E,mask); std::swap(NNF,NNF2);

    checkCudaError(cudaDeviceSynchronize());

    krnlPropagationPass<<<numBlocks,threadsPerBlock>>>(sizeA,sizeB,Omega,patchWidth,patchError,lambda,2,NNF,NNF2,E,mask); std::swap(NNF,NNF2);

    checkCudaError(cudaDeviceSynchronize());

    krnlPropagationPass<<<numBlocks,threadsPerBlock>>>(sizeA,sizeB,Omega,patchWidth,patchError,lambda,1,NNF,NNF2,E,mask); std::swap(NNF,NNF2);

    checkCudaError(cudaDeviceSynchronize());

    krnlRandomSearchPass<<<numBlocks,threadsPerBlock>>>(sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF,E,mask,rngStates);

    checkCudaError(cudaDeviceSynchronize());
  }

  krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchWidth,patchError,NNF,E);

  checkCudaError(cudaDeviceSynchronize());
}

#endif
