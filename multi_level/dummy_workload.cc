#include "dummy_workload.h"
#define GPU_UTIL_FILE "/mnt/ramdisk/gpu_util"
#define GPU_MAT_SIZE 8192 // 655356(25000, PERIOD=30 -> aoubt 22%) 65536(50000, PERIOD=15 -> about 40%) 65536(66535, PERIOD=7 -> about 60%) 65536(66535, PERIOD=2 -> about 80%)
#define need_period
#define cpu_use
#define gpu_use
#define max_cpu 6
#define exp

#define total_exp_time 36
#define single_exp_time 6

//#define hist
int m = 0;

/* Note.MS (24.8.7) 10ms
  65536, x 16, y 16, 5ms - 60%, 0.0095 (NX)
  65536, x 32, y 32, 5ms - 70~90%, 0.034 (NX)
  32768 x 32, y 32, 5ms -  not working (NX) 
  16384 x 32, y 32, 5ms -  not working (NX) 
  8192, x 32, y 32, 5ms - 38~46%, 0.0046 (NX) 
  4096, x 32, y 32, 5ms - 24~30%, 0.0024 (NX) 
  2048, x 32, y 32, 1ms - 36~40%, 0.0014 (NX) 
  2048, x 32, y 32, 5ms - 15~20%, 0.0014 (NX) 
  1024, x 32, y 32, 5ms - (NX) 
  2, x 1, y 1 - (NX)
*/


const char* computeShaderSource = R"(
#version 310 es
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in; // Work group size

layout(std430, binding = 0) readonly buffer MatrixA {
    float A[];
};

layout(std430, binding = 1) readonly buffer MatrixB {
    float B[];
};

layout(std430, binding = 2) writeonly buffer MatrixC {
    float C[];
};

uniform ivec3 sizeA; // (x1, y1, z1)
uniform ivec3 sizeB; // (x2, y2, z2)

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID); // Global ID for each thread

    int x1 = sizeA.x;
    int y1 = sizeA.y;
    int z1 = sizeA.z;

    int x2 = sizeB.x;
    int y2 = sizeB.y;
    int z2 = sizeB.z;

    // Ensure valid multiplication indices
    if (gid.x >= x1 || gid.y >= y2 || gid.z >= z2) {
        return;
    }

    float sum = 0.0;
    for (int i = 0; i < y1; ++i) { // y1 == x2 for matrix multiplication compatibility
        int indexA = gid.x * (y1 * z1) + i * z1 + gid.z;
        int indexB = i * (y2 * z2) + gid.y * z2 + gid.z;
        sum += A[indexA] * B[indexB];
    }

    int indexC = gid.x * (y2 * z2) + gid.y * z2 + gid.z;
    C[indexC] = sum;
}
)";

bool m_break = false;

void INThandler(int sig) {
  signal(sig, SIG_IGN);
  m_break = true;
}

Workload::Workload(){};

Workload::Workload(int duration, float transition_time, int kernel_size,
                   int cpu, bool random) {
  struct timespec init, begin, end, begin_i, end_i;
  
  /* Total execution occurs in duration x size (sec)*/

  int size = 1;
  total_duration = duration;
  cpugpu_transition = transition_time;
  gpu_kernel_size = kernel_size;
  cpu_cores = cpu;

  std::cout << "Dummy workload" << "\n";
  std::cout << "Total duration: " << total_duration << "s \n";
  std::cout << "CPU GPU transition: " << cpugpu_transition << "s \n";
  std::cout << "GPU z2 kernel size: " << gpu_kernel_size << "\n";
  std::cout << "CPU max cores: " << cpu_cores << "\n";

  // int core[11] = {0, 1, 2, 3, 4, 5, 6};
  // int util[11] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  // std::random_device rd;
  // std::mt19937 gen(rd()); // 매 번 정규분포로 뽑아주는 값이 다름 (비교 가능할까?)
  std::mt19937 generator_c; // 매 번 정규분포로 뽑아주는 값이 동일함 (random하다고 볼 수 있을까?)
  std::mt19937 generator_g; // 매 번 정규분포로 뽑아주는 값이 동일함 (random하다고 볼 수 있을까?)
  std::normal_distribution<double> dist_c(/* 평균 = */ 90, /* 표준 편차 = */ 3);
  std::normal_distribution<double> dist_g(/* 평균 = */ 90, /* 표준 편차 = */ 3);
  std::vector<int> hist_c(size);
  std::vector<int> hist_g(size);
  //std::map<int, int> hist_v{};

  for (int n = 0; n < size; ++n) {
    // generator or gen 넣어줘야함
    hist_c[n] = std::round(dist_c(generator_c));
    hist_g[n] = std::round(dist_g(generator_g));
    //++hist_v[hist[n]]; // dist(generator)로 생성한 값(key)의 value를 늘려줌
  }
#ifdef hist
  for (auto p : hist_v) {
    std::cout << std::setw(2) << p.first << ' '
              << std::string(p.second, '*') << " " << p.second << '\n';
  }
#endif
  for (auto p : hist_c) {
    std::cout << p << ' ';
  }
  std::cout << "\n";
  for (auto p : hist_g) {
    std::cout << p << ' ';
  }
  std::cout << "\n";

  clock_gettime(CLOCK_MONOTONIC, &init);
  std::ofstream gpu_util_f, cpu_util_f;
  ////////////// Init 0 0
#ifdef exp
  std::cout << "========Init=========\n";
  ///////////////////////////////////////////////////////////////////////
  ////// workload start 
  double elapsed_t = 0.0;
  double total_elapsed_t = 0.0;

  cpu_workload_pool.reserve(cpu_cores);
  stop = false;
  cpu_worker_termination = false;
  gpu_worker_termination = false;
  for (int i = 0; i < cpu_cores; ++i) {
    std::cout << "Creates " << i << " cpu worker"
              << "\n";
    cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
  }
  //Minsung
  gpu_workload_pool.reserve(1);
  std::cout << "Creates kernel size " << gpu_kernel_size << " workload GPU worker"
            << "\n";
  gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  
  clock_gettime(CLOCK_MONOTONIC, &init);
  while (total_elapsed_t < total_duration) {
    ////////////////////////
    // CPU start (300ms)  //
    ////////////////////////
    cpu_stop = false;
    {  // Wakes CPU workers
        std::unique_lock<std::mutex> lock(cpu_mtx);
        cpu_ignition = true;
        cpu_cv.notify_all();
        std::cout << "Notified CPU workers\n";
    }

    elapsed_t = 0;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    while (elapsed_t < cpugpu_transition) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    }
    printf("CPU elapsed %.6fs\n", elapsed_t);

    total_elapsed_t += elapsed_t;
    printf("CPU total elapsed %.6fs\n", total_elapsed_t);
    std::cout << "CPU workload done\n";

    cpu_stop = true;
    cpu_ignition = false;

    ////////////////////////
    // GPU start (300ms)  //
    ////////////////////////
    gpu_stop = false;
    {  // Wakes GPU workers
        std::unique_lock<std::mutex> lock(gpu_mtx);
        gpu_ignition = true;
        gpu_kernel_done = false;
        gpu_cv.notify_all();
        std::cout << "Notified GPU workers\n";
    }

    // Start GPU timing
    elapsed_t = 0;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    {  // GPU kernel return wait
        std::unique_lock<std::mutex> lock_data(gpu_mtx);
        gpu_end_cv.wait(lock_data, [&] { return gpu_kernel_done; });
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Correct elapsed time calculation after GPU wait
    elapsed_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    printf("After unlock gpu mtx elapsed %.6fs\n", elapsed_t);

    gpu_stop = true;
    total_elapsed_t += elapsed_t;
    std::cout << "GPU workload done\n";
    printf("Total elapsed time: %.6f seconds\n", total_elapsed_t);
}

  // CPU worker kill
  cpu_worker_termination = true;
  cpu_stop = true;
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(cpu_mtx);
    cpu_ignition = true;
    cpu_cv.notify_all();
    std::cout << "Notified all CPU workers to kill"
              << "\n";
  }
  gpu_worker_termination = true;
  gpu_stop = true;
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(gpu_mtx);
    gpu_ignition = true;
    gpu_cv.notify_all();
    std::cout << "Notified GPU workers to kill"
              << "\n";
  }

  // stop = false;
  // {  // wakes  workers
  //   std::unique_lock<std::mutex> lock(mtx);
  //   ignition = true;
  //   cv.notify_all();
  //   std::cout << "Notified all workers"
  //             << "\n";
  // }
  // clock_gettime(CLOCK_MONOTONIC, &begin);
  // elepsed_t = 0;
  //   std::this_thread::sleep_for(std::chrono::milliseconds(100));
  //   clock_gettime(CLOCK_MONOTONIC, &end);
  //   elepsed_t = (end.tv_sec - begin.tv_sec) +
  //               ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  // }
  // // printf("%.6fs\n", elepsed_t);
  // std::cout << "Timeout" << "\n";
  stop = true;
  ignition = false;
  for (auto& workers : gpu_workload_pool) workers.join();
  for (auto& workers : cpu_workload_pool) workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();
  std::cout << "=====================\n";

#endif
};

void Workload::CPU_Worker() {
  // not implemented
  while(!cpu_worker_termination){
    //std::cout << "cpu worker start" << "\n";
    {
      std::unique_lock<std::mutex> lock_(cpu_mtx);
      cpu_cv.wait(lock_, [this]() { return cpu_ignition; });
    }
    double a = 1;
    double b = 0.0003;
    while (!cpu_stop) {
      a *= b;
    }
  }
  std::cout << "Terminates CPU worker " << "\n";
};


void Workload::GPU_Worker() {
    struct timespec init_begin, init_end;
    // Initialize OpenCL platform, device, context, and queue
    clock_gettime(CLOCK_MONOTONIC, &init_begin);
    std::vector<cl_platform_id> platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Replace with actual kernel source
    const char* kernelSource = "computeShaderSource";
    
    // Obtain platform, device, create context and queue
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create and build program and kernel
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);

    cl_context_properties context_properties[] = {
      CL_CONTEXT_PLATFORM, platform,
      0
    };
    
    cl_queue_properties queue_properties[] = {
      0
    };

    cl_int errcode;
    cl_context context = clCreateContext(
      context_properties, 1, &device, nullptr, nullptr, &errcode
    );
    cl_command_queue queue = clCreateCommandQueueWithProperties(
      context, device, queue_properties, &errcode
    );

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, kernelSource, NULL);

    // Allocate and initialize buffers
    const int x1 = 1024, y1 = 128, z1 = 256; // Dimensions for matrix A
    const int x2 = 32, y2 = 32, z2 = gpu_kernel_size;    // Dimensions for matrix B

    std::vector<float> A(x1 * y1 * z1, 1.0f);
    std::vector<float> B(x2 * y2 * z2, 2.0f);
    std::vector<float> C(x1 * y2 * z2, 0.0f);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A.size() * sizeof(float), A.data(), NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(float), B.data(), NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, C.size() * sizeof(float), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &x1);
    clSetKernelArg(kernel, 4, sizeof(int), &y1);
    clSetKernelArg(kernel, 5, sizeof(int), &z1);
    clSetKernelArg(kernel, 6, sizeof(int), &x2);
    clSetKernelArg(kernel, 7, sizeof(int), &y2);
    clSetKernelArg(kernel, 8, sizeof(int), &z2);

    // Define global and local work sizes
    size_t globalSize[3] = { static_cast<size_t>(x1), static_cast<size_t>(y2), static_cast<size_t>(gpu_kernel_size) };
    size_t localSize[3] = { static_cast<size_t>(x1), static_cast<size_t>(y2), static_cast<size_t>(gpu_kernel_size) };

    // Register SIGINT handler
    signal(SIGINT, INThandler);
    clock_gettime(CLOCK_MONOTONIC, &init_end);
    double init_time = (init_end.tv_sec - init_begin.tv_sec) + ((init_end.tv_nsec - init_begin.tv_nsec) / 1000000000.0);
    printf("init time : %.11f\n", init_time);

    struct timespec begin, end;

    while (!gpu_worker_termination) {
      int count = 0;
      double tot_response_t = 0.0, gpu_elapsed_t = 0.0;
      struct timespec seq_begin;

      {
        std::unique_lock<std::mutex> lock_(gpu_mtx);
        gpu_cv.wait(lock_, [this]() { return gpu_ignition; });
      }

      clock_gettime(CLOCK_MONOTONIC, &seq_begin);

      while (!gpu_stop) {
        if (m_break) break;

        clock_gettime(CLOCK_MONOTONIC, &begin);
        clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);
        clFlush(queue);
        clFinish(queue); // Ensure kernel execution completes
        clock_gettime(CLOCK_MONOTONIC, &end);
        double response_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
        printf("gpu %d's response time : %.11f\n", count, response_t);

        gpu_elapsed_t += response_t;
        //  (end.tv_sec - seq_begin.tv_sec) + ((end.tv_nsec - seq_begin.tv_nsec) / 1000000000.0);

        if (gpu_elapsed_t > cpugpu_transition) {
            printf("gpu elapsed time : %.11f\n", gpu_elapsed_t);
            gpu_stop = true;
        }
        count++;
      }
      {
        std::unique_lock<std::mutex> lock_data(gpu_mtx);
        gpu_kernel_done = true;
        gpu_ignition = false;
        gpu_end_cv.notify_one();
      }
      // printf("%d's average : %.11f\n", count, (tot_response_t / double(count)));
    }

    // Read back result from bufferC
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, C.size() * sizeof(float), C.data(), 0, NULL, NULL);

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "GPU worker terminated." << std::endl;

    return;
}

Workload::~Workload(){};
