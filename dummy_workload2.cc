#include "dummy_workload.h"
#define CPU_UTIL_FILE "/home/odroid/TfLite_apps/scheduler/cpu_util"
#define GPU_UTIL_FILE "/home/odroid/TfLite_apps/scheduler/gpu_util"
#define GPU_MAT_SIZE 256  // 64, 128, 256, 512
#define GPU_LOCAL_SIZE 8
#define need_period
#define cpu_use
#define gpu_use
#define exp
//#define hist
int m = 0;

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

bool m_break = false;

void INThandler(int sig) {
  signal(sig, SIG_IGN);
  m_break = true;
}

Workload::Workload(){};

Workload::Workload(int duration, int cpu, int gpu, bool random) {
  struct timespec init, begin, end, begin_i, end_i;
  int range[9] = {0, 1, 2, 1, 0, 1, 2, 1, 0};
  int core[11] = {0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8};
  int util[11] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  // std::random_device rd;
  // std::mt19937 gen(rd()); // 매 번 정규분포로 뽑아주는 값이 다름 (비교 가능할까?)
  std::mt19937 generator_c; // 매 번 정규분포로 뽑아주는 값이 동일함 (random하다고 볼 수 있을까?)
  std::mt19937 generator_g; // 매 번 정규분포로 뽑아주는 값이 동일함 (random하다고 볼 수 있을까?)
  std::normal_distribution<double> dist_c(/* 평균 = */ 60, /* 표준 편차 = */ 13);
  std::normal_distribution<double> dist_g(/* 평균 = */ 80, /* 표준 편차 = */ 13);
  int size = 12;
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
  std::cout << "Got cpu " << cpu << " gpu " << gpu << " duration " << duration
            << "\n";
  ////////////// Init 0 0
#ifdef exp
  std::cout << "========Init=========\n";
  cpu = core[0];
  duration = 2;
  
  cpu_util_f.open(CPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!cpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  if (cpu > 0) {
    cpu_workload_pool.reserve(cpu);
    for (int i = 0; i < cpu; ++i) {
      std::cout << "Creates " << i << " cpu worker" << "\n";
      cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
    }
  }
  cpu_util_f << cpu * 100 << "\n";
  cpu_util_f.close();
  
  gpu = 0;
  gpu_util_f.open(GPU_UTIL_FILE, std::ios::out | std::ios::trunc);
  if (!gpu_util_f.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return;
  }
  if(gpu > 0){
    gpu_workload_pool.reserve(1);
    std::cout << "Creates " << gpu << "% gpu workers"
              << "\n";
    gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  }
  gpu_util_f << gpu << "\n";
  gpu_util_f.close();

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
  while (elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  printf("%.6fs\n", elepsed_t);

  stop = true;
  ignition = false;

  for (auto& workers : cpu_workload_pool) workers.join();
  for (auto& workers : gpu_workload_pool) workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();
  ///////////////////////////////////////////////////////////////////////
  ////// workload start 
  std::cout << "=====================\n";
  int tmp_c, tmp_g;
  duration = 5;
  // clock_gettime(CLOCK_MONOTONIC, &begin_i);
  for(int k=0; k<hist_c.size(); k++){
    tmp_c = hist_c[k];
    tmp_g = hist_g[k];

    if(tmp_c<0) tmp_c *= -1;
    if(tmp_g<0) tmp_g *= -1;
    
    if(tmp_c%10 == 0){
      if(tmp_c > 80) cpu = 8;
      else cpu = tmp_c / 10;
    } 
    else {
      if(tmp_c > 80) cpu = 8;
      else cpu = tmp_c/10 + 1;
    }

    if(tmp_g%10 == 0){
      gpu = tmp_g / 10;
      m = tmp_g / 10;
    } 
    else {
      gpu = tmp_g / 10 + 1;
      m = tmp_g / 10 + 1;
    }

    std::cout << "cpu " << cpu << " gpu" << gpu << "\n";
    cpu_workload_pool.reserve(cpu);
    stop = false;
    for (int i = 0; i < cpu; ++i) {
      std::cout << "Creates " << i << " cpu worker"
                << "\n";
      cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
    }

    gpu_workload_pool.reserve(1);
    std::cout << "Creates " << gpu * 10 << "% gpu workers"
              << "\n";
    gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
    
    cpu_util_f.open(CPU_UTIL_FILE, std::ios::out | std::ios::trunc);
    if (!cpu_util_f.is_open()) {
      std::cerr << "Failed to open cpu_util(h)" << std::endl;
      return;
    }
    gpu_util_f.open(GPU_UTIL_FILE, std::ios::out | std::ios::trunc);
    if (!gpu_util_f.is_open()) {
      std::cerr << "Failed to open gpu_util(h)" << std::endl;
      return;
    }

    cpu_util_f << cpu * 100 << "\n";
    cpu_util_f.close();
    gpu_util_f << gpu * 10 << "\n";
    gpu_util_f.close();

    stop = false;
    {  // wakes  workers
      std::unique_lock<std::mutex> lock(mtx);
      ignition = true;
      cv.notify_all();
      std::cout << "Notified all workers"
                << "\n";
    }
    clock_gettime(CLOCK_MONOTONIC, &begin);
    elepsed_t = 0;
    // std::cout << duration << "\n";
    while (elepsed_t < duration) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      clock_gettime(CLOCK_MONOTONIC, &end);
      elepsed_t = (end.tv_sec - begin.tv_sec) +
                  ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    }
    printf("%.6fs\n", elepsed_t);
    std::cout << "Timeout"
              << "\n";
    stop = true;
    ignition = false;
    for (auto& workers : gpu_workload_pool) workers.join();
    for (auto& workers : cpu_workload_pool) workers.join();
    cpu_workload_pool.clear();
    gpu_workload_pool.clear();
    std::cout << "=====================\n";
  }
  ///////////////////////////////////////////////////////////////////////
  ////////////// Init 0 0
  std::cout << "=========Init========\n";
  cpu = 0;
  gpu = 0;
  duration = 2;

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

  cpu_util_f << 0 << "\n";
  cpu_util_f.close();
  gpu_util_f << 0 << "\n";
  gpu_util_f.close();

  std::this_thread::sleep_for(std::chrono::seconds(1));
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers"
              << "\n";
  }

  elepsed_t = 0;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  while (elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  printf("%.6fs\n", elepsed_t);
  stop = true;
  ignition = false;
  for (auto& workers : gpu_workload_pool) workers.join();
  for (auto& workers : cpu_workload_pool) workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();
  std::cout << "=====================\n";
  //////////////////////////////////////////////////////////////////////
  std::cout << "Dummy workload end"
            << "\n";
#endif
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
  int period[11] = {0, 100, 50, 30, 20, 14, 10, 5, 3, 1, 0};  // kernel period (HZ)
  signal(SIGINT, INThandler);
  std::ofstream outfile;
  while (!stop) {
    if (m_break) break;
    int PERIOD = period[m];

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
        cl::NDRange(GPU_LOCAL_SIZE, GPU_LOCAL_SIZE), NULL, NULL);

    queue.finish();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);

    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0,
                            sizeof(float) * matrixElements,
                            resultMatrix.data());

    // 매핑 해제
    // clock_gettime(CLOCK_MONOTONIC, &begin);
    queue.enqueueUnmapMemObject(bufferA, mapped_ptr_A);
    queue.enqueueUnmapMemObject(bufferB, mapped_ptr_B);
 
    count++;
    total_elapsed_t = 0;
  }
}

Workload::~Workload(){};
