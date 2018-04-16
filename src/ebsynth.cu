// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#include "ebsynth.h"
#include "patchmatch_gpu.h"

#define FOR(A,X,Y) for(int Y=0;Y<A.height();Y++) for(int X=0;X<A.width();X++)

A2V2i nnfInitRandom(const V2i& targetSize,
                    const V2i& sourceSize,
                    const int  patchSize)
{
  A2V2i NNF(targetSize);
  const int r = patchSize/2;

  for (int i = 0; i < NNF.numel(); i++)
  {
      NNF[i] = V2i
      (
          r+(rand()%(sourceSize[0]-2*r)),
          r+(rand()%(sourceSize[1]-2*r))
      );
  }

  return NNF;
}

A2V2i nnfUpscale(const A2V2i& NNF,
                 const int    patchSize,
                 const V2i&   targetSize,
                 const V2i&   sourceSize)
{
  A2V2i NNF2x(targetSize);

  FOR(NNF2x,x,y)
  {
    NNF2x(x,y) = NNF(clamp(x/2,0,NNF.width()-1),
                     clamp(y/2,0,NNF.height()-1))*2+V2i(x%2,y%2);
  }

  FOR(NNF2x,x,y)
  {
    const V2i nn = NNF2x(x,y);

    NNF2x(x,y) = V2i(clamp(nn(0),patchSize,sourceSize(0)-patchSize-1),
                     clamp(nn(1),patchSize,sourceSize(1)-patchSize-1));
  }

  return NNF2x;
}

template<int N, typename T, int M>
__global__ void krnlVotePlain(      TexArray2<N,T,M> target,
                              const TexArray2<N,T,M> source,
                              const TexArray2<2,int> NNF,
                              const int              patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<target.width && y<target.height)
  {
    const int r = patchSize / 2;

    Vec<N,float> sumColor = zero<Vec<N,float>>::value();
    float sumWeight = 0;

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      /*
      if
      (
        x+px >= 0 && x+px < NNF.width () &&
        y+py >= 0 && y+py < NNF.height()
      )
      */
      {
        const V2i n = NNF(x+px,y+py)-V2i(px,py);

        /*if
        (
          n[0] >= 0 && n[0] < S.width () &&
          n[1] >= 0 && n[1] < S.height()
        )*/
        {
          const float weight = 1.0f;
          sumColor += weight*Vec<N,float>(source(n(0),n(1)));
          sumWeight += weight;
        }
      }
    }

    const Vec<N,T> v = Vec<N,T>(sumColor/sumWeight);
    target.write(x,y,v);
  }
}

template<int N, typename T, int M>
__global__ void krnlVoteWeighted(      TexArray2<N,T,M>   target,
                                 const TexArray2<N,T,M>   source,
                                 const TexArray2<2,int>   NNF,
                                 const TexArray2<1,float> E,
                                 const int patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<target.width && y<target.height)
  {
    const int r = patchSize / 2;

    Vec<N,float> sumColor = zero<Vec<N,float>>::value();
    float sumWeight = 0;

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      /*
      if
      (
        x+px >= 0 && x+px < NNF.width () &&
        y+py >= 0 && y+py < NNF.height()
      )
      */
      {
        const V2i n = NNF(x+px,y+py)-V2i(px,py);

        /*if
        (
          n[0] >= 0 && n[0] < S.width () &&
          n[1] >= 0 && n[1] < S.height()
        )*/
        {
          const float error = E(x+px,y+py)(0)/(patchSize*patchSize*N);
          const float weight = 1.0f/(1.0f+error);
          sumColor += weight*Vec<N,float>(source(n(0),n(1)));
          sumWeight += weight;
        }
      }
    }

    const Vec<N,T> v = Vec<N,T>(sumColor/sumWeight);
    target.write(x,y,v);
  }
}

template<int N, typename T, int M>
__device__ Vec<N,T> sampleBilinear(const TexArray2<N,T,M>& I,float x,float y)
{
  const int ix = x;
  const int iy = y;

  const float s = x-ix;
  const float t = y-iy;

  // XXX: clamp!!!
  return Vec<N,T>((1.0f-s)*(1.0f-t)*Vec<N,float>(I(ix  ,iy  ))+
                  (     s)*(1.0f-t)*Vec<N,float>(I(ix+1,iy  ))+
                  (1.0f-s)*(     t)*Vec<N,float>(I(ix  ,iy+1))+
                  (     s)*(     t)*Vec<N,float>(I(ix+1,iy+1)));
};

template<int N, typename T, int M>
__global__ void krnlResampleBilinear(TexArray2<N,T,M> O,
                                     const TexArray2<N,T,M> I)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<O.width && y<O.height)
  {
    const float s = float(I.width)/float(O.width);
    O.write(x,y,sampleBilinear(I,s*float(x),s*float(y)));
  }
}

template<int N, typename T, int M>
__global__ void krnlEvalMask(      TexArray2<1,unsigned char> mask,
                             const TexArray2<N,T,M> style,
                             const TexArray2<N,T,M> style2,
                             const int stopThreshold)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<mask.width && y<mask.height)
  {
    const Vec<N,T> s  = style(x,y);
    const Vec<N,T> s2 = style2(x,y);

    int maxDiff = 0;
    for(int c=0;c<N;c++)
    {
      const int diff = std::abs(int(s[c])-int(s2[c]));
      maxDiff = diff>maxDiff ? diff:maxDiff;
    }

    const Vec<1,unsigned char> msk = maxDiff < stopThreshold ? Vec<1,unsigned char>(0) : Vec<1,unsigned char>(255);

    mask.write(x,y,msk);
  }
}

__global__ void krnlDilateMask(TexArray2<1,unsigned char> mask2,
                               const TexArray2<1,unsigned char> mask,
                               const int patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<mask.width && y<mask.height)
  {
    const int r = patchSize / 2;

    Vec<1,unsigned char> msk = Vec<1,unsigned char>(0);

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      if (mask(x+px,y+py)[0]==255) { msk = Vec<1,unsigned char>(255); }
    }

    mask2.write(x,y,msk);
  }
}

template<int N, typename T, int M>
void resampleGPU(      TexArray2<N,T,M>& O,
                 const TexArray2<N,T,M>& I)
{
  const int numThreadsPerBlock = 24;
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
  const dim3 numBlocks = dim3((O.width+threadsPerBlock.x)/threadsPerBlock.x,
                              (O.height+threadsPerBlock.y)/threadsPerBlock.y);

  krnlResampleBilinear<<<numBlocks,threadsPerBlock>>>(O,I);

  checkCudaError(cudaDeviceSynchronize());
}

template<int NS,int NG,typename T>
struct PatchSSD_Split
{
  const TexArray2<NS,T> targetStyle;
  const TexArray2<NS,T> sourceStyle;

  const TexArray2<NG,T> targetGuide;
  const TexArray2<NG,T> sourceGuide;

  const Vec<NS,float> styleWeights;
  const Vec<NG,float> guideWeights;

  PatchSSD_Split(const TexArray2<NS,T>& targetStyle,
                 const TexArray2<NS,T>& sourceStyle,

                 const TexArray2<NG,T>& targetGuide,
                 const TexArray2<NG,T>& sourceGuide,

                 const Vec<NS,float>&   styleWeights,
                 const Vec<NG,float>&   guideWeights)

  : targetStyle(targetStyle),sourceStyle(sourceStyle),
    targetGuide(targetGuide),sourceGuide(sourceGuide),
    styleWeights(styleWeights),guideWeights(guideWeights) {}

   __device__ float operator()(const int   patchSize,
                               const int   tx,
                               const int   ty,
                               const int   sx,
                               const int   sy,
                               const float ebest)
  {
    const int r = patchSize/2;
    float error = 0;

    for(int py=-r;py<=+r;py++)
    {
      for(int px=-r;px<=+r;px++)
      {
        {
          const Vec<NS,T> pixTs = targetStyle(tx + px,ty + py);
          const Vec<NS,T> pixSs = sourceStyle(sx + px,sy + py);
          for(int i=0;i<NS;i++)
          {
            const float diff = float(pixTs[i]) - float(pixSs[i]);
            error += styleWeights[i]*diff*diff;
          }
        }

        {
          const Vec<NG,T> pixTg = targetGuide(tx + px,ty + py);
          const Vec<NG,T> pixSg = sourceGuide(sx + px,sy + py);
          for(int i=0;i<NG;i++)
          {
            const float diff = float(pixTg[i]) - float(pixSg[i]);
            error += guideWeights[i]*diff*diff;
          }
        }
      }

      if (error>ebest) { return error; }
    }

    return error;
  }
};

template<int NS,int NG,typename T>
struct PatchSSD_Split_Modulation
{
  const TexArray2<NS,T> targetStyle;
  const TexArray2<NS,T> sourceStyle;

  const TexArray2<NG,T> targetGuide;
  const TexArray2<NG,T> sourceGuide;

  const TexArray2<NG,T> targetModulation;

  const Vec<NS,float> styleWeights;
  const Vec<NG,float> guideWeights;

  PatchSSD_Split_Modulation(const TexArray2<NS,T>& targetStyle,
                            const TexArray2<NS,T>& sourceStyle,

                            const TexArray2<NG,T>& targetGuide,
                            const TexArray2<NG,T>& sourceGuide,

                            const TexArray2<NG,T>& targetModulation,

                            const Vec<NS,float>&   styleWeights,
                            const Vec<NG,float>&   guideWeights)

  : targetStyle(targetStyle),sourceStyle(sourceStyle),
    targetGuide(targetGuide),sourceGuide(sourceGuide),
    targetModulation(targetModulation),
    styleWeights(styleWeights),guideWeights(guideWeights) {}

   __device__ float operator()(const int   patchSize,
                               const int   tx,
                               const int   ty,
                               const int   sx,
                               const int   sy,
                               const float ebest)
  {
    const int r = patchSize/2;
    float error = 0;

    for(int py=-r;py<=+r;py++)
    {
      for(int px=-r;px<=+r;px++)
      {
        {
          const Vec<NS,T> pixTs = targetStyle(tx + px,ty + py);
          const Vec<NS,T> pixSs = sourceStyle(sx + px,sy + py);
          for(int i=0;i<NS;i++)
          {
            const float diff = float(pixTs[i]) - float(pixSs[i]);
            error += styleWeights[i]*diff*diff;
          }
        }

        {
          const Vec<NG,T> pixTg = targetGuide(tx + px,ty + py);
          const Vec<NG,T> pixSg = sourceGuide(sx + px,sy + py);
          const Vec<NG,float> mult = Vec<NG,float>(targetModulation(tx,ty))/255.0f;

          for(int i=0;i<NG;i++)
          {
            const float diff = float(pixTg[i]) - float(pixSg[i]);
            error += guideWeights[i]*mult[i]*diff*diff;
          }
        }
      }

      if (error>ebest) { return error; }
    }

    return error;
  }
};

V2i pyramidLevelSize(const V2i& sizeBase,const int numLevels,const int level)
{
  return V2i(V2f(sizeBase)*pow(2.0f,-float(numLevels-1-level)));
}

template<int NS,int NG>
void runEbsynth(int    ebsynthBackend,
                int    numStyleChannels,
                int    numGuideChannels,
                int    sourceWidth,
                int    sourceHeight,
                void*  sourceStyleData,
                void*  sourceGuideData,
                int    targetWidth,
                int    targetHeight,
                void*  targetGuideData,
                void*  targetModulationData,
                float* styleWeights,
                float* guideWeights,
                float  uniformityWeight,
                int    patchSize,
                int    voteMode,
                int    numPyramidLevels,
                int*   numSearchVoteItersPerLevel,
                int*   numPatchMatchItersPerLevel,
                int*   stopThresholdPerLevel,
                void*  outputData)
{
  const int levelCount = numPyramidLevels;

  struct PyramidLevel
  {
    PyramidLevel() { }

    int sourceWidth;
    int sourceHeight;
    int targetWidth;
    int targetHeight;

    TexArray2<NS,unsigned char> sourceStyle;
    TexArray2<NG,unsigned char> sourceGuide;
    TexArray2<NS,unsigned char> targetStyle;
    TexArray2<NS,unsigned char> targetStyle2;
    TexArray2<1,unsigned char>  mask;
    TexArray2<1,unsigned char>  mask2;
    TexArray2<NG,unsigned char> targetGuide;
    TexArray2<NG,unsigned char> targetModulation;
    TexArray2<2,int>            NNF;
    TexArray2<2,int>            NNF2;
    TexArray2<1,float>          E;
    MemArray2<int>              Omega;
  };

  std::vector<PyramidLevel> pyramid(levelCount);
  for(int level=0;level<levelCount;level++)
  {
    const V2i levelSourceSize = pyramidLevelSize(V2i(sourceWidth,sourceHeight),levelCount,level);
    const V2i levelTargetSize = pyramidLevelSize(V2i(targetWidth,targetHeight),levelCount,level);

    pyramid[level].sourceWidth  = levelSourceSize(0);
    pyramid[level].sourceHeight = levelSourceSize(1);
    pyramid[level].targetWidth  = levelTargetSize(0);
    pyramid[level].targetHeight = levelTargetSize(1);

    pyramid[level].sourceStyle  = TexArray2<NS,unsigned char>(levelSourceSize);
    pyramid[level].sourceGuide  = TexArray2<NG,unsigned char>(levelSourceSize);
    pyramid[level].targetStyle  = TexArray2<NS,unsigned char>(levelTargetSize);
    pyramid[level].targetStyle2 = TexArray2<NS,unsigned char>(levelTargetSize);
    pyramid[level].mask         = TexArray2<1,unsigned char>(levelTargetSize);
    pyramid[level].mask2        = TexArray2<1,unsigned char>(levelTargetSize);
    pyramid[level].targetGuide  = TexArray2<NG,unsigned char>(levelTargetSize);
    pyramid[level].NNF          = TexArray2<2,int>  (levelTargetSize);
    pyramid[level].NNF2         = TexArray2<2,int>  (levelTargetSize);
    pyramid[level].E            = TexArray2<1,float>(levelTargetSize);
    pyramid[level].Omega        = MemArray2<int>    (levelSourceSize);

    if (targetModulationData) { pyramid[level].targetModulation = TexArray2<NG,unsigned char>(levelTargetSize); }
  }

  copy(&pyramid[levelCount-1].sourceStyle,sourceStyleData);
  copy(&pyramid[levelCount-1].sourceGuide,sourceGuideData);
  copy(&pyramid[levelCount-1].targetGuide,targetGuideData);
  if (targetModulationData) { copy(&pyramid[levelCount-1].targetModulation,targetModulationData); }

  for(int level=0;level<levelCount-1;level++)
  {
    resampleGPU(pyramid[level].sourceStyle,pyramid[levelCount-1].sourceStyle);
    resampleGPU(pyramid[level].sourceGuide,pyramid[levelCount-1].sourceGuide);
    resampleGPU(pyramid[level].targetGuide,pyramid[levelCount-1].targetGuide);
    if (targetModulationData) { resampleGPU(pyramid[level].targetModulation,pyramid[levelCount-1].targetModulation); }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  bool inExtraPass = false;

  curandState* rngStates = initGpuRng(targetWidth,targetHeight);

  for (int level=0;level<pyramid.size();level++)
  {
    /////////////////////////////////////////////////////////////////////////////

    if (!inExtraPass)
    {
      A2V2i cpu_NNF;
      if (level>0)
      {
        A2V2i prevLevelNNF(pyramid[level-1].targetWidth,
                           pyramid[level-1].targetHeight);

        copy(&prevLevelNNF,pyramid[level-1].NNF);

        cpu_NNF = nnfUpscale(prevLevelNNF,
                             patchSize,
                             V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                             V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight));
      }
      else
      {
        cpu_NNF = nnfInitRandom(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                                V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                                patchSize);
      }
      copy(&pyramid[level].NNF,cpu_NNF);

      /////////////////////////////////////////////////////////////////////////
      Array2<int> cpu_Omega(pyramid[level].sourceWidth,pyramid[level].sourceHeight);

      fill(&cpu_Omega,(int)0);
      for(int ay=0;ay<cpu_NNF.height();ay++)
      for(int ax=0;ax<cpu_NNF.width();ax++)
      {
        const V2i& n = cpu_NNF(ax,ay);
        const int bx = n(0);
        const int by = n(1);

        const int r = patchSize/2;

        for(int oy=-r;oy<=+r;oy++)
        for(int ox=-r;ox<=+r;ox++)
        {
          const int x = bx+ox;
          const int y = by+oy;
          cpu_Omega(x,y) += 1;
        }
      }

      copy(&pyramid[level].Omega,cpu_Omega);
      /////////////////////////////////////////////////////////////////////////
    }

    ////////////////////////////////////////////////////////////////////////////
    {
      const int numThreadsPerBlock = 24;
      const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
      const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                  (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

      krnlVotePlain<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                   pyramid[level].sourceStyle,
                                                   pyramid[level].NNF,
                                                   patchSize);

      std::swap(pyramid[level].targetStyle2,pyramid[level].targetStyle);
      checkCudaError( cudaDeviceSynchronize() );
    }
    ////////////////////////////////////////////////////////////////////////////

    Array2<Vec<1,unsigned char>> cpu_mask(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight));
    fill(&cpu_mask,Vec<1,unsigned char>(255));
    copy(&pyramid[level].mask,cpu_mask);

    ////////////////////////////////////////////////////////////////////////////

    for (int voteIter=0;voteIter<numSearchVoteItersPerLevel[level];voteIter++)
    {
      Vec<NS,float> styleWeightsVec;
      for(int i=0;i<NS;i++) { styleWeightsVec[i] = styleWeights[i]; }

      Vec<NG,float> guideWeightsVec;
      for(int i=0;i<NG;i++) { guideWeightsVec[i] = guideWeights[i]; }

      const int numGpuThreadsPerBlock = 24;

      if (numPatchMatchItersPerLevel[level]>0)
      {
        if (targetModulationData)
        {
          patchmatchGPU(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                        V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                        pyramid[level].Omega,
                        patchSize,
                        PatchSSD_Split_Modulation<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                                       pyramid[level].sourceStyle,
                                                                       pyramid[level].targetGuide,
                                                                       pyramid[level].sourceGuide,
                                                                       pyramid[level].targetModulation,
                                                                       styleWeightsVec,
                                                                       guideWeightsVec),
                        uniformityWeight,
                        numPatchMatchItersPerLevel[level],
                        numGpuThreadsPerBlock,
                        pyramid[level].NNF,
                        pyramid[level].NNF2,
                        pyramid[level].E,
                        pyramid[level].mask,
                        rngStates);
        }
        else
        {
          patchmatchGPU(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                        V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                        pyramid[level].Omega,
                        patchSize,
                        PatchSSD_Split<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                            pyramid[level].sourceStyle,
                                                            pyramid[level].targetGuide,
                                                            pyramid[level].sourceGuide,
                                                            styleWeightsVec,
                                                            guideWeightsVec),
                        uniformityWeight,
                        numPatchMatchItersPerLevel[level],
                        numGpuThreadsPerBlock,
                        pyramid[level].NNF,
                        pyramid[level].NNF2,
                        pyramid[level].E,
                        pyramid[level].mask,
                        rngStates);
        }
      }
      else
      {
        const int numThreadsPerBlock = 24;
        const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
        const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                    (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

        if (targetModulationData)
        {
          krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchSize,
                                                           PatchSSD_Split_Modulation<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                                                                          pyramid[level].sourceStyle,
                                                                                                          pyramid[level].targetGuide,
                                                                                                          pyramid[level].sourceGuide,
                                                                                                          pyramid[level].targetModulation,
                                                                                                          styleWeightsVec,
                                                                                                          guideWeightsVec),
                                                           pyramid[level].NNF,
                                                           pyramid[level].E);
        }
        else
        {
          krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchSize,
                                                           PatchSSD_Split<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                                                               pyramid[level].sourceStyle,
                                                                                               pyramid[level].targetGuide,
                                                                                               pyramid[level].sourceGuide,
                                                                                               styleWeightsVec,
                                                                                               guideWeightsVec),
                                                           pyramid[level].NNF,
                                                           pyramid[level].E);
        }
        checkCudaError( cudaDeviceSynchronize() );
      }

      {
        const int numThreadsPerBlock = 24;
        const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
        const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                    (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

        if      (voteMode==EBSYNTH_VOTEMODE_PLAIN)
        {
          krnlVotePlain<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                       pyramid[level].sourceStyle,
                                                       pyramid[level].NNF,
                                                       patchSize);
        }
        else if (voteMode==EBSYNTH_VOTEMODE_WEIGHTED)
        {
          krnlVoteWeighted<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                          pyramid[level].sourceStyle,
                                                          pyramid[level].NNF,
                                                          pyramid[level].E,
                                                          patchSize);
        }

        std::swap(pyramid[level].targetStyle2,pyramid[level].targetStyle);
        checkCudaError( cudaDeviceSynchronize() );

        if (voteIter<numSearchVoteItersPerLevel[level]-1)
        {
          krnlEvalMask<<<numBlocks,threadsPerBlock>>>(pyramid[level].mask,
                                                      pyramid[level].targetStyle,
                                                      pyramid[level].targetStyle2,
                                                      stopThresholdPerLevel[level]);
          checkCudaError( cudaDeviceSynchronize() );

          krnlDilateMask<<<numBlocks,threadsPerBlock>>>(pyramid[level].mask2,
                                                        pyramid[level].mask,
                                                        patchSize);
          std::swap(pyramid[level].mask2,pyramid[level].mask);
          checkCudaError( cudaDeviceSynchronize() );
        }
      }
    }
  }

  checkCudaError( cudaDeviceSynchronize() );

  copy(&outputData,pyramid[pyramid.size()-1].targetStyle);

  checkCudaError( cudaFree(rngStates) );

  for(int level=0;level<pyramid.size();level++)
  {
    pyramid[level].sourceStyle.destroy();
    pyramid[level].sourceGuide.destroy();
    pyramid[level].targetStyle.destroy();
    pyramid[level].targetStyle2.destroy();
    pyramid[level].mask.destroy();
    pyramid[level].mask2.destroy();
    pyramid[level].targetGuide.destroy();
    pyramid[level].NNF.destroy();
    pyramid[level].NNF2.destroy();
    pyramid[level].E.destroy();
    pyramid[level].Omega.destroy();
    if (targetModulationData) { pyramid[level].targetModulation.destroy(); }
  }
}

EBSYNTH_API void ebsynthRun(int    ebsynthBackend,
                            int    numStyleChannels,
                            int    numGuideChannels,
                            int    sourceWidth,
                            int    sourceHeight,
                            void*  sourceStyleData,
                            void*  sourceGuideData,
                            int    targetWidth,
                            int    targetHeight,
                            void*  targetGuideData,
                            void*  targetModulationData,
                            float* styleWeights,
                            float* guideWeights,
                            float  uniformityWeight,
                            int    patchSize,
                            int    voteMode,
                            int    numPyramidLevels,
                            int*   numSearchVoteItersPerLevel,
                            int*   numPatchMatchItersPerLevel,
                            int*   stopThresholdPerLevel,
                            void*  outputData
                            )
{
  void (*const dispatchEbsynth[EBSYNTH_MAX_GUIDE_CHANNELS][EBSYNTH_MAX_STYLE_CHANNELS])(int,int,int,int,int,void*,void*,int,int,void*,void*,float*,float*,float,int,int,int,int*,int*,int*,void*) =
  {
    { runEbsynth<1, 1>, runEbsynth<2, 1>, runEbsynth<3, 1>, runEbsynth<4, 1>, runEbsynth<5, 1>, runEbsynth<6, 1>, runEbsynth<7, 1>, runEbsynth<8, 1> },
    { runEbsynth<1, 2>, runEbsynth<2, 2>, runEbsynth<3, 2>, runEbsynth<4, 2>, runEbsynth<5, 2>, runEbsynth<6, 2>, runEbsynth<7, 2>, runEbsynth<8, 2> },
    { runEbsynth<1, 3>, runEbsynth<2, 3>, runEbsynth<3, 3>, runEbsynth<4, 3>, runEbsynth<5, 3>, runEbsynth<6, 3>, runEbsynth<7, 3>, runEbsynth<8, 3> },
    { runEbsynth<1, 4>, runEbsynth<2, 4>, runEbsynth<3, 4>, runEbsynth<4, 4>, runEbsynth<5, 4>, runEbsynth<6, 4>, runEbsynth<7, 4>, runEbsynth<8, 4> },
    { runEbsynth<1, 5>, runEbsynth<2, 5>, runEbsynth<3, 5>, runEbsynth<4, 5>, runEbsynth<5, 5>, runEbsynth<6, 5>, runEbsynth<7, 5>, runEbsynth<8, 5> },
    { runEbsynth<1, 6>, runEbsynth<2, 6>, runEbsynth<3, 6>, runEbsynth<4, 6>, runEbsynth<5, 6>, runEbsynth<6, 6>, runEbsynth<7, 6>, runEbsynth<8, 6> },
    { runEbsynth<1, 7>, runEbsynth<2, 7>, runEbsynth<3, 7>, runEbsynth<4, 7>, runEbsynth<5, 7>, runEbsynth<6, 7>, runEbsynth<7, 7>, runEbsynth<8, 7> },
    { runEbsynth<1, 8>, runEbsynth<2, 8>, runEbsynth<3, 8>, runEbsynth<4, 8>, runEbsynth<5, 8>, runEbsynth<6, 8>, runEbsynth<7, 8>, runEbsynth<8, 8> },
    { runEbsynth<1, 9>, runEbsynth<2, 9>, runEbsynth<3, 9>, runEbsynth<4, 9>, runEbsynth<5, 9>, runEbsynth<6, 9>, runEbsynth<7, 9>, runEbsynth<8, 9> },
    { runEbsynth<1,10>, runEbsynth<2,10>, runEbsynth<3,10>, runEbsynth<4,10>, runEbsynth<5,10>, runEbsynth<6,10>, runEbsynth<7,10>, runEbsynth<8,10> },
    { runEbsynth<1,11>, runEbsynth<2,11>, runEbsynth<3,11>, runEbsynth<4,11>, runEbsynth<5,11>, runEbsynth<6,11>, runEbsynth<7,11>, runEbsynth<8,11> },
    { runEbsynth<1,12>, runEbsynth<2,12>, runEbsynth<3,12>, runEbsynth<4,12>, runEbsynth<5,12>, runEbsynth<6,12>, runEbsynth<7,12>, runEbsynth<8,12> },
    { runEbsynth<1,13>, runEbsynth<2,13>, runEbsynth<3,13>, runEbsynth<4,13>, runEbsynth<5,13>, runEbsynth<6,13>, runEbsynth<7,13>, runEbsynth<8,13> },
    { runEbsynth<1,14>, runEbsynth<2,14>, runEbsynth<3,14>, runEbsynth<4,14>, runEbsynth<5,14>, runEbsynth<6,14>, runEbsynth<7,14>, runEbsynth<8,14> },
    { runEbsynth<1,15>, runEbsynth<2,15>, runEbsynth<3,15>, runEbsynth<4,15>, runEbsynth<5,15>, runEbsynth<6,15>, runEbsynth<7,15>, runEbsynth<8,15> },
    { runEbsynth<1,16>, runEbsynth<2,16>, runEbsynth<3,16>, runEbsynth<4,16>, runEbsynth<5,16>, runEbsynth<6,16>, runEbsynth<7,16>, runEbsynth<8,16> },
    { runEbsynth<1,17>, runEbsynth<2,17>, runEbsynth<3,17>, runEbsynth<4,17>, runEbsynth<5,17>, runEbsynth<6,17>, runEbsynth<7,17>, runEbsynth<8,17> },
    { runEbsynth<1,18>, runEbsynth<2,18>, runEbsynth<3,18>, runEbsynth<4,18>, runEbsynth<5,18>, runEbsynth<6,18>, runEbsynth<7,18>, runEbsynth<8,18> },
    { runEbsynth<1,19>, runEbsynth<2,19>, runEbsynth<3,19>, runEbsynth<4,19>, runEbsynth<5,19>, runEbsynth<6,19>, runEbsynth<7,19>, runEbsynth<8,19> },
    { runEbsynth<1,20>, runEbsynth<2,20>, runEbsynth<3,20>, runEbsynth<4,20>, runEbsynth<5,20>, runEbsynth<6,20>, runEbsynth<7,20>, runEbsynth<8,20> },
    { runEbsynth<1,21>, runEbsynth<2,21>, runEbsynth<3,21>, runEbsynth<4,21>, runEbsynth<5,21>, runEbsynth<6,21>, runEbsynth<7,21>, runEbsynth<8,21> },
    { runEbsynth<1,22>, runEbsynth<2,22>, runEbsynth<3,22>, runEbsynth<4,22>, runEbsynth<5,22>, runEbsynth<6,22>, runEbsynth<7,22>, runEbsynth<8,22> },
    { runEbsynth<1,23>, runEbsynth<2,23>, runEbsynth<3,23>, runEbsynth<4,23>, runEbsynth<5,23>, runEbsynth<6,23>, runEbsynth<7,23>, runEbsynth<8,23> },
    { runEbsynth<1,24>, runEbsynth<2,24>, runEbsynth<3,24>, runEbsynth<4,24>, runEbsynth<5,24>, runEbsynth<6,24>, runEbsynth<7,24>, runEbsynth<8,24> }
  };

  if (numStyleChannels>=1 && numStyleChannels<=EBSYNTH_MAX_STYLE_CHANNELS &&
      numGuideChannels>=1 && numGuideChannels<=EBSYNTH_MAX_GUIDE_CHANNELS)
  {
    dispatchEbsynth[numGuideChannels-1][numStyleChannels-1](ebsynthBackend,
                                                            numStyleChannels,
                                                            numGuideChannels,
                                                            sourceWidth,
                                                            sourceHeight,
                                                            sourceStyleData,
                                                            sourceGuideData,
                                                            targetWidth,
                                                            targetHeight,
                                                            targetGuideData,
                                                            targetModulationData,
                                                            styleWeights,
                                                            guideWeights,
                                                            uniformityWeight,
                                                            patchSize,
                                                            voteMode,
                                                            numPyramidLevels,
                                                            numSearchVoteItersPerLevel,
                                                            numPatchMatchItersPerLevel,
                                                            stopThresholdPerLevel,
                                                            outputData);
  }
}

EBSYNTH_API
int ebsynthBackendAvailable(int ebsynthBackend)
{
  if (ebsynthBackend==EBSYNTH_BACKEND_CUDA)
  {
    int deviceCount = -1;
    if (cudaGetDeviceCount(&deviceCount)!=cudaSuccess) { return 0; }

    for (int device=0;device<deviceCount;device++)
    {
      cudaDeviceProp properties;
      if (cudaGetDeviceProperties(&properties,device)==cudaSuccess)
      {
        if (properties.major!=9999 && properties.major>=3)
        {
          return 1;
        }
      }
    }
  }

  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cmath>

#include <vector>
#include <string>
#include <algorithm>

#include "jzq.h"

template<typename FUNC>
bool tryToParseArg(const std::vector<std::string>& args,int* inout_argi,const char* name,bool* out_fail,FUNC handler)
{
  int& argi = *inout_argi;
  bool& fail = *out_fail;

  if (argi<0 || argi>=args.size()) { fail = true; return false; }

  if (args[argi]==name)
  {
    argi++;
    fail = !handler();    
    return true;
  }

  fail = false; return false; 
}

bool tryToParseIntArg(const std::vector<std::string>& args,int* inout_argi,const char* name,int* out_value,bool* out_fail)
{
  return tryToParseArg(args,inout_argi,name,out_fail,[&]
  {
    int& argi = *inout_argi;
    if (argi<args.size())
    {
      const std::string& arg = args[argi];
      try
      {
        std::size_t pos = 0;
        *out_value = std::stoi(arg,&pos);
        if (pos!=arg.size()) { printf("error: bad %s argument '%s'\n",name,arg.c_str()); return false; }
        return true;
      }
      catch(...)
      {
        printf("error: bad %s argument '%s'\n",name,arg.c_str());
        return false;
      }   
    }
    printf("error: missing argument for the %s option\n",name);
    return false;
  });
}

bool tryToParseFloatArg(const std::vector<std::string>& args,int* inout_argi,const char* name,float* out_value,bool* out_fail)
{
  return tryToParseArg(args,inout_argi,name,out_fail,[&]
  {
    int& argi = *inout_argi;
    if (argi<args.size())
    {
      const std::string& arg = args[argi];
      try
      {
        std::size_t pos = 0;
        *out_value = std::stof(arg,&pos);
        if (pos!=arg.size()) { printf("error: bad %s argument '%s'\n",name,arg.c_str()); return false; }
        return true;
      }
      catch(...)
      {
        printf("error: bad %s argument '%s'\n",name,args[argi].c_str());
        return false;
      }   
    }
    printf("error: missing argument for the %s option\n",name);
    return false;
  });
}

bool tryToParseStringArg(const std::vector<std::string>& args,int* inout_argi,const char* name,std::string* out_value,bool* out_fail)
{
  return tryToParseArg(args,inout_argi,name,out_fail,[&]
  {
    int& argi = *inout_argi;
    if (argi<args.size())
    {
      *out_value = args[argi];
      return true;
    }
    printf("error: missing argument for the %s option\n",name);
    return false;
  });
}

bool tryToParseStringPairArg(const std::vector<std::string>& args,int* inout_argi,const char* name,std::pair<std::string,std::string>* out_value,bool* out_fail)
{
  return tryToParseArg(args,inout_argi,name,out_fail,[&]
  {
    int& argi = *inout_argi;
    if ((argi+1)<args.size())
    {
      *out_value = std::make_pair(args[argi],args[argi+1]);
      argi++;
      return true;
    }
    printf("error: missing argument for the %s option\n",name);
    return false;
  });
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

unsigned char* tryLoad(const std::string& fileName,int* width,int* height)
{
  unsigned char* data = stbi_load(fileName.c_str(),width,height,NULL,4);
  if (data==NULL)
  {
    printf("error: failed to load '%s'\n",fileName.c_str());
    printf("%s\n",stbi_failure_reason());
    exit(1);
  }
  return data;
}

int evalNumChannels(const unsigned char* data,const int numPixels)
{
  bool isGray = true;
  bool hasAlpha = false;

  for(int xy=0;xy<numPixels;xy++)
  {
    const unsigned char r = data[xy*4+0];
    const unsigned char g = data[xy*4+1];
    const unsigned char b = data[xy*4+2];
    const unsigned char a = data[xy*4+3];

    if (!(r==g && g==b)) { isGray  = false; }
    if (a<255)           { hasAlpha = true; }
  }

  const int numChannels = (isGray ? 1 : 3) + (hasAlpha ? 1 : 0);

  return numChannels;
}

V2i pyramidLevelSize(const V2i& sizeBase,const int level)
{
  return V2i(V2f(sizeBase)*pow(2.0f,-float(level)));
}

int main(int argc,char** argv)
{
  if (argc<2)
  {
    printf("usage: %s [options]\n",argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -style <style.png>\n");
    printf("  -guide <source.png> <target.png>\n");
    printf("  -output <output.png>\n");
    printf("  -weight <value>\n");
    printf("  -uniformity <value>\n");
    printf("  -patchsize <size>\n");
    printf("  -pyramidlevels <number>\n");
    printf("  -searchvoteiters <number>\n");
    printf("  -patchmatchiters <number>\n");
    printf("  -stopthreshold <value>\n");
    printf("\n");
    return 1;
  }

  std::string styleFileName;
  float       styleWeight = NAN;
  std::string outputFileName = "output.png";

  struct Guide
  {
    std::string    sourceFileName;
    std::string    targetFileName;
    float          weight;

    int            sourceWidth;
    int            sourceHeight;
    unsigned char* sourceData;

    int            targetWidth;
    int            targetHeight;
    unsigned char* targetData;
    
    int            numChannels;
  };

  std::vector<Guide> guides;

  float uniformityWeight = 3500;
  int patchSize = 5; 
  int numPyramidLevels = -1;
  int numSearchVoteIters = 6;
  int numPatchMatchIters = 4;
  int stopThreshold = 5;

  std::string backend;

  {
    std::vector<std::string> args(argc);
    for(int i=0;i<argc;i++) { args[i] = argv[i]; }
  
    bool fail = false;
    int argi = 1;   

    float* precedingStyleOrGuideWeight = 0;
    while(argi<argc && !fail)
    {
      float weight;
      std::pair<std::string,std::string> guidePair;
      
      if      (tryToParseStringArg(args,&argi,"-style",&styleFileName,&fail))
      {
        styleWeight = NAN;
        precedingStyleOrGuideWeight = &styleWeight;
        argi++;
      }
      else if (tryToParseStringPairArg(args,&argi,"-guide",&guidePair,&fail))
      {
        Guide guide;
        guide.sourceFileName = guidePair.first;
        guide.targetFileName = guidePair.second;
        guide.weight = NAN;
        guides.push_back(guide);
        precedingStyleOrGuideWeight = &guides[guides.size()-1].weight;
        argi++;
      }
      else if (tryToParseStringArg(args,&argi,"-output",&outputFileName,&fail))
      {
        argi++;
      }
      else if (tryToParseFloatArg(args,&argi,"-weight",&weight,&fail))
      {
        if (precedingStyleOrGuideWeight!=0) { *precedingStyleOrGuideWeight = weight; }
        else { printf("error: at least one -style or -guide option must precede the -weight option!\n"); return 1; }
        argi++;
      }
      else if (tryToParseFloatArg(args,&argi,"-uniformity",&uniformityWeight,&fail)) { argi++; }
      else if (tryToParseIntArg(args,&argi,"-patchsize",&patchSize,&fail))
      {
        if (patchSize<3)    { printf("error: patchsize is too small!\n"); return 1; }
        if (patchSize%2==0) { printf("error: patchsize must be an odd number!\n"); return 1; }
        argi++;
      }
      else if (tryToParseIntArg(args,&argi,"-pyramidlevels",&numPyramidLevels,&fail))
      {
        if (numPyramidLevels<1) { printf("error: bad argument for -pyramidlevels!\n"); return 1; }
        argi++;
      }
      else if (tryToParseIntArg(args,&argi,"-searchvoteiters",&numSearchVoteIters,&fail))
      {
        if (numSearchVoteIters<0) { printf("error: bad argument for -searchvoteiters!\n"); return 1; }
        argi++;
      }
      else if (tryToParseIntArg(args,&argi,"-patchmatchiters",&numPatchMatchIters,&fail))
      {
        if (numPatchMatchIters<0) { printf("error: bad argument for -patchmatchiters!\n"); return 1; }
        argi++;
      }
      else if (tryToParseIntArg(args,&argi,"-stopthreshold",&stopThreshold,&fail))
      {
        if (stopThreshold<0) { printf("error: bad argument for -stopthreshold!\n"); return 1; }
        argi++;
      }
      else
      {
        printf("error: unrecognized option '%s'\n",args[argi].c_str());
        fail = true;
      }
    }
    
    if (fail) { return 1; }
  }

  const int numGuides = guides.size();

  int sourceWidth = 0;
  int sourceHeight = 0;
  unsigned char* sourceStyleData = tryLoad(styleFileName,&sourceWidth,&sourceHeight);
  const int numStyleChannelsTotal = evalNumChannels(sourceStyleData,sourceWidth*sourceHeight);

  std::vector<unsigned char> sourceStyle(sourceWidth*sourceHeight*numStyleChannelsTotal);
  for(int xy=0;xy<sourceWidth*sourceHeight;xy++)
  {
    if      (numStyleChannelsTotal>0)  { sourceStyle[xy*numStyleChannelsTotal+0] = sourceStyleData[xy*4+0]; }
    if      (numStyleChannelsTotal==2) { sourceStyle[xy*numStyleChannelsTotal+1] = sourceStyleData[xy*4+3]; }           
    else if (numStyleChannelsTotal>1)  { sourceStyle[xy*numStyleChannelsTotal+1] = sourceStyleData[xy*4+1]; }
    if      (numStyleChannelsTotal>2)  { sourceStyle[xy*numStyleChannelsTotal+2] = sourceStyleData[xy*4+2]; }
    if      (numStyleChannelsTotal>3)  { sourceStyle[xy*numStyleChannelsTotal+3] = sourceStyleData[xy*4+3]; }                 
  }
  
  int targetWidth = 0;
  int targetHeight = 0;
  int numGuideChannelsTotal = 0;

  for(int i=0;i<numGuides;i++)
  {
    Guide& guide = guides[i];

    guide.sourceData = tryLoad(guide.sourceFileName,&guide.sourceWidth,&guide.sourceHeight);
    guide.targetData = tryLoad(guide.targetFileName,&guide.targetWidth,&guide.targetHeight);
      
    if              (guide.sourceWidth!=sourceWidth || guide.sourceHeight!=sourceHeight)  { printf("error: source guide '%s' doesn't match the resolution of '%s'\n",guide.sourceFileName.c_str(),styleFileName.c_str()); return 1; }      
    if      (i>0 && (guide.targetWidth!=targetWidth || guide.targetHeight!=targetHeight)) { printf("error: target guide '%s' doesn't match the resolution of '%s'\n",guide.targetFileName.c_str(),guides[0].targetFileName.c_str()); return 1; }
    else if (i==0) { targetWidth = guide.targetWidth; targetHeight = guide.targetHeight; }

    guide.numChannels = std::max(evalNumChannels(guide.sourceData,sourceWidth*sourceHeight),
                                 evalNumChannels(guide.targetData,targetWidth*targetHeight));    
  
    numGuideChannelsTotal += guide.numChannels;
  }
  
  if (numStyleChannelsTotal>EBSYNTH_MAX_STYLE_CHANNELS) { printf("error: too many style channels (%d), maximum number is %d\n",numStyleChannelsTotal,EBSYNTH_MAX_STYLE_CHANNELS); return 1; }
  if (numGuideChannelsTotal>EBSYNTH_MAX_GUIDE_CHANNELS) { printf("error: too many guide channels (%d), maximum number is %d\n",numGuideChannelsTotal,EBSYNTH_MAX_GUIDE_CHANNELS); return 1; }

  std::vector<unsigned char> sourceGuides(sourceWidth*sourceHeight*numGuideChannelsTotal);
  for(int xy=0;xy<sourceWidth*sourceHeight;xy++)
  {
    int c = 0;
    for(int i=0;i<numGuides;i++)
    { 
      const int numChannels = guides[i].numChannels;  

      if      (numChannels>0)  { sourceGuides[xy*numGuideChannelsTotal+c+0] = guides[i].sourceData[xy*4+0]; }
      if      (numChannels==2) { sourceGuides[xy*numGuideChannelsTotal+c+1] = guides[i].sourceData[xy*4+3]; }           
      else if (numChannels>1)  { sourceGuides[xy*numGuideChannelsTotal+c+1] = guides[i].sourceData[xy*4+1]; }
      if      (numChannels>2)  { sourceGuides[xy*numGuideChannelsTotal+c+2] = guides[i].sourceData[xy*4+2]; }
      if      (numChannels>3)  { sourceGuides[xy*numGuideChannelsTotal+c+3] = guides[i].sourceData[xy*4+3]; }            
      
      c += numChannels;
    }
  }

  std::vector<unsigned char> targetGuides(targetWidth*targetHeight*numGuideChannelsTotal);
  for(int xy=0;xy<targetWidth*targetHeight;xy++)
  {
    int c = 0;
    for(int i=0;i<numGuides;i++)
    { 
      const int numChannels = guides[i].numChannels;  

      if      (numChannels>0)  { targetGuides[xy*numGuideChannelsTotal+c+0] = guides[i].targetData[xy*4+0]; }
      if      (numChannels==2) { targetGuides[xy*numGuideChannelsTotal+c+1] = guides[i].targetData[xy*4+3]; }           
      else if (numChannels>1)  { targetGuides[xy*numGuideChannelsTotal+c+1] = guides[i].targetData[xy*4+1]; }
      if      (numChannels>2)  { targetGuides[xy*numGuideChannelsTotal+c+2] = guides[i].targetData[xy*4+2]; }
      if      (numChannels>3)  { targetGuides[xy*numGuideChannelsTotal+c+3] = guides[i].targetData[xy*4+3]; }            
      
      c += numChannels;
    }
  }

  std::vector<float> styleWeights(numStyleChannelsTotal);
  if (isnan(styleWeight)) { styleWeight = 1.0f; }
  for(int i=0;i<numStyleChannelsTotal;i++) { styleWeights[i] = styleWeight / float(numStyleChannelsTotal); }

  for(int i=0;i<numGuides;i++) { if (isnan(guides[i].weight)) { guides[i].weight = 1.0f/float(numGuides); } }

  std::vector<float> guideWeights(numGuideChannelsTotal);
  {
    int c = 0;
    for(int i=0;i<numGuides;i++)
    { 
      const int numChannels = guides[i].numChannels;  
      
      for(int j=0;j<numChannels;j++)
      {
        guideWeights[c+j] = guides[i].weight / float(numChannels);
      }

      c += numChannels; 
    }
  }

  int maxPyramidLevels = 0;
  for(int level=32;level>=0;level--)
  {
    if (min(pyramidLevelSize(std::min(V2i(sourceWidth,sourceHeight),V2i(targetWidth,targetHeight)),level)) >= (2*patchSize+1))
    {
      maxPyramidLevels = level+1;
      break;
    }
  }

  if (numPyramidLevels==-1) { numPyramidLevels = maxPyramidLevels; }
  numPyramidLevels = std::min(numPyramidLevels,maxPyramidLevels); 

  std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
  std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
  std::vector<int> stopThresholdPerLevel(numPyramidLevels);
  for(int i=0;i<numPyramidLevels;i++)
  {
    numSearchVoteItersPerLevel[i] = numSearchVoteIters;
    numPatchMatchItersPerLevel[i] = numPatchMatchIters;
    stopThresholdPerLevel[i] = stopThreshold;
  }

  std::vector<unsigned char> output(targetWidth*targetHeight*numStyleChannelsTotal);

  printf("uniformity: %.0f\n",uniformityWeight);
  printf("patchsize: %d\n",patchSize);
  printf("pyramidlevels: %d\n",numPyramidLevels);
  printf("searchvoteiters: %d\n",numSearchVoteIters);
  printf("patchmatchiters: %d\n",numPatchMatchIters);
  printf("stopthreshold: %d\n",stopThreshold);

  if (!ebsynthBackendAvailable(EBSYNTH_BACKEND_CUDA)) { printf("error: the CUDA backend is not available!\n"); return 1; }

  ebsynthRun(EBSYNTH_BACKEND_CUDA,
             numStyleChannelsTotal,
             numGuideChannelsTotal,
             sourceWidth,
             sourceHeight,
             sourceStyle.data(),
             sourceGuides.data(),
             targetWidth,
             targetHeight,
             targetGuides.data(),
             NULL,
             styleWeights.data(),
             guideWeights.data(),
             uniformityWeight,
             patchSize,
             EBSYNTH_VOTEMODE_PLAIN,
             numPyramidLevels,
             numSearchVoteItersPerLevel.data(),
             numPatchMatchItersPerLevel.data(),
             stopThresholdPerLevel.data(),
             output.data());

  stbi_write_png(outputFileName.c_str(),targetWidth,targetHeight,numStyleChannelsTotal,output.data(),numStyleChannelsTotal*targetWidth);

  printf("result was written to %s\n",outputFileName.c_str());

  stbi_image_free(sourceStyleData);

  for(int i=0;i<numGuides;i++)
  {
    stbi_image_free(guides[i].sourceData);
    stbi_image_free(guides[i].targetData);
  }
  
  return 0;
}
