#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImageUtil.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

//smooth out the potential to avoid divide by zero errors
#define EPS2 0.00001 

//GPU parameters
const int nThreads=1024;

/*Number of particles to skip in input. nSkip=1 
  gives 100,000 particles. nSkip=10 gives 10,000 
  particles. Overall 100,000/nSkip total particles. */
const int nSkip=10;

//Produce only nFrames worth of animation. 10 seconds at 30fps is 300 frames, which is a good number for homeworks.
const int nFrames=30;



// GPU Gems style HANDLE_ERROR function
static void HandleError(cudaError_t err,
        const char *file,
        int line) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


//Some string manipulation functions for saving files. pad_int(1234,5) returns "01234". 
std::string pad_int(int arg, int padcount) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(padcount) << arg;
    return ss.str();
}

//Returns a file name in the form of "prefix00###suffix". For example "image0032.bmp"
std::string getFilename(std::string prefix, int num, int padcount, std::string suffix) {
    return prefix + pad_int(num, padcount) + suffix;
}


//Hybrid of GPU Gems 3 ch. 31 and CUDA nbody example.
//20 FLOPs
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r;
    // r_ij  [3 FLOPs]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPs]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f / sqrtf(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    // a_i =  a_i + s * r_ij [6 FLOPs]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

/* Hybrid of GPU Gems 3 ch. 31 and samples/5_Simulations/nbody/bodyststemcuda.cu */
__device__ float4 calculate_acceleration(float4 *devX, int numParticles, int deviceOffset) {
    __shared__ float4 shPosition[nThreads];
    float4 myPosition;
    int tile;
    float3 acc = { 0.0f, 0.0f, 0.0f };
    int gtid = deviceOffset + blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = devX[gtid];
    for (tile = 0; tile<numParticles / blockDim.x; tile++) {
        shPosition[threadIdx.x] = devX[tile * blockDim.x + threadIdx.x];
        __syncthreads();
#pragma unroll 64
        for (unsigned int j = 0; j < blockDim.x; j++) {
            acc = bodyBodyInteraction(myPosition, shPosition[j], acc);
        }
        __syncthreads();
    }
    float4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
    return acc4;
}

__global__ void kernel_step(float4 *devX, float4 *devV, float dt, int numParticles, int deviceOffset) {
    int index = deviceOffset + blockIdx.x * blockDim.x + threadIdx.x;

    //6 FLOPs
    devX[index].x += devV[index].x*dt;
    devX[index].y += devV[index].y*dt;
    devX[index].z += devV[index].z*dt;

    float4 acc = calculate_acceleration(devX, numParticles, deviceOffset);

    //6 FLOPs.
    devV[index].x += acc.x*dt;
    devV[index].y += acc.y*dt;
    devV[index].z += acc.z*dt;
}


int main()
{
    int nParticles;

    //Particle loading code! 
    std::vector<float4> particlepos;
    std::vector<float4> particlevel;
    //from http://stackoverflow.com/a/8421315/1030718
    std::ifstream source;                    // build a read-Stream
    source.open("galaxy1.txt", std::ios_base::in);  // open data
    int ctr=-1;
    for (std::string line; std::getline(source, line); )   //read stream line by line
    {
        ctr++;
        std::istringstream in(line);      //make a stream for the line itself
        if(ctr%nSkip!=0)
            continue;
        float m;
        float x;
        float y;
        float z;
        float vx;
        float vy;
        float vz;
        in >> m >> x >> y >> z >> vx >> vy >> vz;
        //multiplying my nSkip makes sure that if we only have one particle for every 10 particles in the data file, it has 10 times the mass.
        particlepos.push_back(make_float4(x,y,z,m*nSkip ));
        particlevel.push_back(make_float4( vx,vy,vz,0.0 ));
    }
    nParticles = (int)particlepos.size();
    //ensure nParticles is a multiple of nThreads*deviceCount
    int nParticlesNew = nThreads*int(float(nParticles) / (nThreads));
    if (nParticles != nParticlesNew) {
        std::cout << "WARNING in GPU_Phy::initialize. Position list argument size is not a multiple of nThreads*deviceCount!" << std::endl;
        std::cout << "Continuing by chopping to nearest multiple." << std::endl;
        nParticles = nParticlesNew; //this rounds down. nParticles<=positions.size().
    }

    float4 *host_x = new float4[nParticles];
    float4 *host_v = new float4[nParticles];
    for (int i = 0; i < nParticles; i++) {
        host_x[i] = particlepos[i];
        host_v[i] = particlevel[i];
    }

    int nBlocks = nParticles / (nThreads);

    HANDLE_ERROR(cudaSetDevice(0));

    //initialize device memory
    float4 *dev_x;
    float4 *dev_v;

    //Allocate the memory on the device
    HANDLE_ERROR(cudaMalloc((void**)(&dev_x), nParticles*sizeof(float4)));
    HANDLE_ERROR(cudaMalloc((void**)(&dev_v), nParticles*sizeof(float4)));

    //Copy the memory to the device
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, nParticles*sizeof(float4), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_v, host_v, nParticles*sizeof(float4), cudaMemcpyHostToDevice));

    //timestep and save pictures
    float dt = 0.01;
    for (int i = 0; i < nFrames; i++) {
        std::cout << i << std::endl;
        //Do the computation
        for (int k = 0; k < 15; k++){
            kernel_step <<<nBlocks, nThreads >>>(dev_x, dev_v, dt, nParticles, 0);
            cudaDeviceSynchronize();
        }

        //wait for the computation to finish
        cudaDeviceSynchronize();

        //copy the memory to the computer
        HANDLE_ERROR(cudaMemcpy(host_x, dev_x, nParticles*sizeof(float4), cudaMemcpyDeviceToHost));

        //construct an image.
        //First fill up two arrays of scalars representing the particle density.
        //One for the dark matter (which will be reddish), and one for regular matter (bluish).
        DoubleImage regularMass(400, 400);
        DoubleImage darkMatter(400, 400);
        Image pic(400,400);
        for (int j = 0; j < nParticles; j++) {
            int x = int(host_x[j].x * 20) + 400 / 2;
            int y = int(host_x[j].y * 20) + 400 / 2;
            if(j<50000/nSkip){
                regularMass.increase(x, y, 1);
                regularMass.increase(x + 1, y, 0.5);
                regularMass.increase(x - 1, y, 0.5);
                regularMass.increase(x, y + 1, 0.5);
                regularMass.increase(x, y - 1, 0.5);
            } else {
                darkMatter.increase(x, y, 1);
                darkMatter.increase(x + 1, y, 0.5);
                darkMatter.increase(x - 1, y, 0.5);
                darkMatter.increase(x, y + 1, 0.5);
                darkMatter.increase(x, y - 1, 0.5);
            }
        }
        //Combine the two scalar arrays into an array of rgb values.
        for (int x = 0; x < 400; x++) {
            for (int y = 0; y < 400; y++) {
                //Human eye sees brightness logarithmically, so take a log.
                double scalar1 = log(1+3*regularMass.get(x, y))/4.0 ;
                double scalar2 = log(1 + darkMatter.get(x, y)) / 3.0;
                //combine everything into a nice color.
                pic.put(x, y, floatToRGB(scalar2*1.2+scalar1, 0+scalar1*1.5, scalar1*1.7));
            }
        }
        //actually save the image
        pic.save(getFilename("out/image", i, 3, ".bmp"));

    }


    delete [] host_x;
    delete [] host_v;
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_v));
    cudaDeviceReset();

    return 0;
}
