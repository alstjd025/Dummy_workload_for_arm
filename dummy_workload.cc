#include "dummy_workload.h"
#define CPU_UTIL_FILE "/home/odroid/TfLite_apps/scheduler/cpu_util"
#define GPU_UTIL_FILE "/home/odroid/TfLite_apps/scheduler/gpu_util"
#define GPU_MAT_SIZE 256
#define GPU_LOCAL_SIZE 8
// #define PERIOD 20
//  10Hz -> 100,50Hz -> 20, 100Hz -> 10
// #define need_period

const char* kernelSource = R"(
    __kernel void matrixMultiply(__global float* A, __global float* B,
    __global float* C, const int N) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        float acc = 0;
        for (int k=0; k<N; k++)
            acc += A[i*N + k] * B[k*N + j];
        C[i*N + j] = acc;
    }
)";

Workload::Workload(){};

Workload::Workload(int duration, int cpu, int gpu, bool random) {
  struct timespec init, begin, end;
  clock_gettime(CLOCK_MONOTONIC, &init);
  std::ofstream gpu_util_f, cpu_util_f;
  std::cout << "Got cpu " << cpu << " gpu " << gpu << " duration " << duration
            << "\n";
  stop = false;
  // if(gpu > 0){
  //   gpu_workload_pool.reserve(gpu);
  //   for(int i=0; i<gpu; ++i){
  //     std::cout << "Creates " << i << " gpu worker" << "\n";
  //     gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  //   }
  // }

  ///////// CPU 0 GPU 0
  cpu_util_f.open(CPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!cpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  gpu_util_f.open(GPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!gpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }

  cpu = 0;
  duration = 2;
  if (cpu > 0) {
    cpu_workload_pool.reserve(cpu);
    for (int i = 0; i < cpu; ++i) {
      // std::cout << "Creates " << i << " cpu worker" << "\n";
      cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
    }
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers"
              << "\n";
  }

  double elepsed_t = 0;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  elepsed_t = (begin.tv_sec - init.tv_sec) +
              ((begin.tv_nsec - init.tv_nsec) / 1000000000.0);
  printf("start time : %.6f \n", elepsed_t);
  std::cout << duration << "\n";
  while (elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  // std::cout << "Timeout" << "\n";
  stop = true;
  ignition = false;
  for (auto& workers : gpu_workload_pool) workers.join();
  for (auto& workers : cpu_workload_pool) workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();

  ///////// CPU 400 GPU 0
  cpu = 4;
  duration = 10;
  cpu_workload_pool.reserve(cpu);
  stop = false;
  for (int i = 0; i < cpu; ++i) {
    // std::cout << "Creates " << i << " cpu worker" << "\n";
    cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
  }
  cpu_util_f << "400"
             << "\n";
  cpu_util_f.close();
  gpu_util_f << "0"
             << "\n";
  gpu_util_f.close();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers"
              << "\n";
  }
  clock_gettime(CLOCK_MONOTONIC, &begin);
  elepsed_t = 0;
  std::cout << duration << "\n";
  while (elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  std::cout << "Timeout"
            << "\n";
  stop = true;
  ignition = false;
  for (auto& workers : gpu_workload_pool) workers.join();
  for (auto& workers : cpu_workload_pool) workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();

  // ///////// CPU 0 GPU 100
  gpu_workload_pool.reserve(1);
  cpu_util_f.open(CPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!cpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  gpu_util_f.open(GPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!gpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  for (int i = 0; i < 1; ++i) {
    std::cout << "Creates " << i << " gpu worker"
              << "\n";
    gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  }
  cpu_util_f << "0"
             << "\n";
  cpu_util_f.close();
  gpu_util_f << "100"
             << "\n";
  gpu_util_f.close();

  stop = false;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers"
              << "\n";
  }
  clock_gettime(CLOCK_MONOTONIC, &begin);
  elepsed_t = 0;
  std::cout << duration << "\n";
  while (elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  std::cout << "Timeout"
            << "\n";
  stop = true;
  ignition = false;
  for (auto& workers : gpu_workload_pool) workers.join();
  for (auto& workers : cpu_workload_pool) workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();

  ///////// CPU 200 GPU 100
  cpu_util_f.open(CPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!cpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  gpu_util_f.open(GPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!gpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  cpu = 2;
  cpu_workload_pool.reserve(cpu);
  stop = false;
  for (int i = 0; i < cpu; ++i) {
    std::cout << "Creates " << i << " cpu worker"
              << "\n";
    cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
  }
  gpu_workload_pool.reserve(1);
  for (int i = 0; i < 1; ++i) {
    std::cout << "Creates " << i << " gpu worker"
              << "\n";
    gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  }
  cpu_util_f << "200"
             << "\n";
  cpu_util_f.close();
  gpu_util_f << "100"
             << "\n";
  gpu_util_f.close();

  stop = false;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers"
              << "\n";
  }
  clock_gettime(CLOCK_MONOTONIC, &begin);
  elepsed_t = 0;
  std::cout << duration << "\n";
  while (elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  std::cout << "Timeout"
            << "\n";
  stop = true;
  ignition = false;
  for (auto& workers : gpu_workload_pool) workers.join();
  for (auto& workers : cpu_workload_pool) workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();
  cpu_util_f.open(CPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!cpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  gpu_util_f.open(GPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!gpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  cpu_util_f << "0"
             << "\n";
  cpu_util_f.close();
  gpu_util_f << "0"
             << "\n";
  gpu_util_f.close();
  std::cout << "Dummy workload end"
            << "\n";
};

void Workload::CPU_Worker() {
  // not implemented
  // std::cout << "Created new CPU worker \n";
  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }
  double a = 1;
  double b = 0.0003;
  while (!stop) {
    a *= b;
  }
  // std::cout << "Terminates CPU worker " << "\n";
};

void Workload::GPU_Worker() {
  int count = 1;
  double elapsed_t, total_elapsed_t = 0;
  struct timespec begin, end;
  int idx;
  // std::string file_name = "latency.txt";

  // Set up OpenCL context, device, and queue
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  ////////////////Select a platform
  auto platform = platforms.front();
  // Create a device
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  // Select a device
  auto device = devices.front();

  // Create a context
  cl::Context context(device);

  // Create a command
  cl::CommandQueue queue(context, device);

  // Compile the OpenCL kernel
  cl::Program program(context, kernelSource);

  program.build("-cl-std=CL1.2");
  // Initialize matrices and create buffers
  const int matrixElements = GPU_MAT_SIZE * GPU_MAT_SIZE;
  std::vector<float> matrixA(matrixElements);
  std::vector<float> matrixB(matrixElements);
  std::vector<float> resultMatrix(matrixElements);

  for (int i = 0; i < matrixElements; ++i) {
    matrixA[i] = static_cast<float>(i);
    matrixB[i] = static_cast<float>(i + matrixElements);
    resultMatrix[i] = static_cast<float>(0);
  }

  // For verification
  // for (int i = 0; i < matrixElements; i++) {
  //   if (i % GPU_MAT_SIZE == 0) std::cout << "\n";
  //   std::cout << matrixA[i] << " ";
  // }
  // std::cout << "\n";
  // for (int i = 0; i < matrixElements; i++) {
  //   if (i % GPU_MAT_SIZE == 0) std::cout << "\n";
  //   std::cout << matrixB[i] << " ";
  // }
  // std::cout << "\n";

  // std::cout << std::fixed;
  // std::cout << std::setprecision(0);
  // // For verification
  // for (int i = 0; i < matrixElements; i++) {
  //   if (i % GPU_MAT_SIZE == 0) std::cout << "\n";
  //   std::cout << matC[i] << " ";
  // }
  // std::cout << "\n";

  // Initialize the arrays...
  cl::Buffer bufferA(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE,
                     matrixA.data());
  cl::Buffer bufferB(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * GPU_MAT_SIZE * GPU_MAT_SIZE,
                     matrixB.data());
  cl::Buffer bufferResult(context, CL_MEM_READ_WRITE,
                          sizeof(float) * matrixElements);

  queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * matrixElements,
                           matrixA.data());
  queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * matrixElements,
                           matrixB.data());

  // Set kernel arguments
  cl::Kernel kernel(program, "matrixMultiply");
  kernel.setArg(0, bufferA);
  kernel.setArg(1, bufferB);
  kernel.setArg(2, bufferResult);
  kernel.setArg(3, GPU_MAT_SIZE);

  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }
  // Launch kernel and measure execution time

  while (!stop) {
    cl::Event* event;
    if (event == nullptr) {
      event = new cl::Event();
    }
    void* mapped_ptr_A = queue.enqueueMapBuffer(
        bufferA, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * matrixElements);
    void* mapped_ptr_B = queue.enqueueMapBuffer(
        bufferB, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * matrixElements);

#ifdef need_period
    std::this_thread::sleep_for(std::chrono::milliseconds(PERIOD));
#endif
    clock_gettime(CLOCK_MONOTONIC, &begin);
    queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(GPU_MAT_SIZE, GPU_MAT_SIZE),
        cl::NDRange(GPU_LOCAL_SIZE, GPU_LOCAL_SIZE), NULL, event);
    queue.finish();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    total_elapsed_t += elapsed_t;
    // printf("%d's time: %.11f\n", count, elapsed_t);
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0,
                            sizeof(float) * matrixElements,
                            resultMatrix.data());
    // // 매핑 해제
    queue.enqueueUnmapMemObject(bufferA, mapped_ptr_A);
    queue.enqueueUnmapMemObject(bufferB, mapped_ptr_B);
    // std::ofstream outfile(file_name);

    // if (!outfile.is_open()) {
    //   std::cerr << "Failed to open " << file_name << std::endl;
    //   return;
    // }

    // outfile << std::fixed << std::setprecision(11);
    // outfile << elapsed_t;
    // outfile << std::endl;
    count++;
  }
  // printf("%d's average time: %.11f\n", count - 1,
  //        total_elapsed_t / (count - 1));
  if (idx == GPU_MAT_SIZE) {
    printf("Verification success!\n");
  }
}

Workload::~Workload(){};
