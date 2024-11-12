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
    __kernel void matmul(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const unsigned int M,
        const unsigned int N,
        const unsigned int P) {

        int row = get_global_id(0);
        int col = get_global_id(1);

        if (row < M && col < P) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * P + col];
            }
            C[row * P + col] = sum;
        }
    })";


bool m_break = false;

void INThandler(int sig) {
  signal(sig, SIG_IGN);
  m_break = true;
}

Workload::Workload(){};

void WriteUtilization(int utilization){
  std::ofstream gpu_util_file("gpu_util");
    if (gpu_util_file.is_open()) {
        gpu_util_file << utilization;
        gpu_util_file.close();
    } else {
        std::cerr << "Failed to open gpu_util file for writing.\n";
    }
    std::cout << "[EZE] GPU util : " << utilization << std::endl;
}


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
  std::cout << "gpu created\n";
  clock_gettime(CLOCK_MONOTONIC, &init);
  while (total_elapsed_t < total_duration) {
    ////////////////////////
    // CPU start (300ms)  //
    ////////////////////////

    //////////////////////// EZE
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
        // WriteUtilization(0);
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
    printf("\033[0;33mGPU START\033[0m\n");
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
    // WriteUtilization(100);
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
    std::cout << "cpu worker start" << "\n";
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
    clock_gettime(CLOCK_MONOTONIC, &init_begin);

    try {
        // OpenCL 플랫폼, 디바이스, 큐 설정
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");

        auto platform = platforms.front();
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) throw std::runtime_error("No GPU devices found on the platform.");

        auto device = devices.front();
        std::cout << "Successfully retrieved platform and device.\n";

        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, computeShaderSource);

        program.build("-cl-std=CL1.2");
        std::cout << "OpenCL kernel compiled successfully.\n";

        // 버퍼 및 행렬 크기 초기화
        const int x1 = 1024, y1 = 128, z1 = 256;
        const int x2 = 32, y2 = 32, z2 = gpu_kernel_size;
        const int matrixElements = x1 * y2 * z2;
        std::vector<float> matrixA(x1 * y1 * z1);
        std::vector<float> matrixB(matrixElements);
        std::vector<float> resultMatrix(matrixElements);

        for (int i = 0; i < matrixElements; ++i) {
            matrixA[i] = static_cast<float>(i);
            matrixB[i] = static_cast<float>(i + matrixElements);
            resultMatrix[i] = static_cast<float>(0);
        }

        // 버퍼 할당
        cl::Buffer bufferA(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * x1 * y1 * z1, matrixA.data());
        cl::Buffer bufferB(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * x2 * y2 * z2, matrixB.data());
        cl::Buffer bufferResult(context, CL_MEM_READ_WRITE, sizeof(float) * matrixElements);
        
        cl::Kernel kernel(program, "matmul");  // 커널 이름을 matmul로 변경
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferResult);
        kernel.setArg(3, x1);  // M (행 수)
        kernel.setArg(4, y1);  // N (내적 크기)
        kernel.setArg(5, z2);  // P (열 수)

        // 버퍼 및 커널 인수 설정 확인
        std::cout << "Buffers and kernel arguments initialized successfully.\n";

        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * x1 * y1 * z1, matrixA.data());
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * x2 * y2 * z2, matrixB.data());

        signal(SIGINT, INThandler);
        clock_gettime(CLOCK_MONOTONIC, &init_end);
        double init_time = (init_end.tv_sec - init_begin.tv_sec) + ((init_end.tv_nsec - init_begin.tv_nsec) / 1e9);
        printf("init time : %.11f\n", init_time);

        struct timespec begin, end;
        std::cout << "Ready to perform matrix multiplication.\n";

        while (!gpu_worker_termination) {
            int count = 0;
            double tot_response_t = 0.0, gpu_elapsed_t = 0.0;
            struct timespec seq_begin;

            void* mapped_ptr_A = queue.enqueueMapBuffer(bufferA, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * x1 * y1 * z1);
            if (mapped_ptr_A == nullptr) throw std::runtime_error("Failed to map buffer A.");

            void* mapped_ptr_B = queue.enqueueMapBuffer(bufferB, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * x2 * y2 * z2);
            if (mapped_ptr_B == nullptr) throw std::runtime_error("Failed to map buffer B.");

            {
                std::unique_lock<std::mutex> lock_(gpu_mtx);
                gpu_cv.wait(lock_, [this]() { return gpu_ignition; });
            }

            clock_gettime(CLOCK_MONOTONIC, &seq_begin);

            while (!gpu_stop) {
                if (m_break) break;

                clock_gettime(CLOCK_MONOTONIC, &begin);
                // 커널 실행: 행렬 곱셈을 위한 NDRange 설정
                cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(x1, z2), cl::NDRange(1, 1), NULL, NULL);
                std::cout << err << std::endl;
                if (err != CL_SUCCESS) {
                    std::cerr << "Failed to enqueue NDRange kernel, error code: " << err << "\n";
                    throw std::runtime_error("Kernel execution failed.");
                }
                queue.finish();

                clock_gettime(CLOCK_MONOTONIC, &end);
                double response_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1e9);
                gpu_elapsed_t += response_t;
                printf("response_t time : %.11f\n", response_t);
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

            queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, sizeof(float) * matrixElements, resultMatrix.data());
            std::cout << "Result matrix successfully read back from GPU.\n";

            queue.enqueueUnmapMemObject(bufferA, mapped_ptr_A);
            queue.enqueueUnmapMemObject(bufferB, mapped_ptr_B);
            std::cout << "Buffers unmapped successfully.\n";
        }

        std::cout << "GPU worker terminated.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}



Workload::~Workload(){};

