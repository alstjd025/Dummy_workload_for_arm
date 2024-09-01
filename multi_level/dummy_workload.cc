#include "dummy_workload.h"
#define CPU_UTIL_FILE "/home/nvidia/TfLite_apps/scheduler/cpu_util"
#define GPU_UTIL_FILE "/home/nvidia/TfLite_apps/scheduler/gpu_util"
#define GPU_MAT_SIZE 8192 // 655356(25000, PERIOD=30 -> aoubt 22%) 65536(50000, PERIOD=15 -> about 40%) 65536(66535, PERIOD=7 -> about 60%) 65536(66535, PERIOD=2 -> about 80%)
#define need_period
#define cpu_use
#define gpu_use
#define exp
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

    layout (local_size_x = 32, local_size_y = 32) in;

    layout (std140, binding = 0) buffer InputMatrixA {
        float matrixA[];
    } inputMatrixA;

    layout (std140, binding = 1) buffer InputMatrixB {
        float matrixB[];
    } inputMatrixB;

    layout (std140, binding = 2) buffer OutputMatrix {
        float resultMatrix[];
    } outputMatrix;

    void main() {
        ivec2 idx = ivec2(gl_GlobalInvocationID.xy);
        float sum = 0.0;
        for (int k = 0; k < 8192; ++k) {
            sum += inputMatrixA.matrixA[idx.y * 8192 + k] * inputMatrixB.matrixB[k * 8192 + idx.x];
        }
        outputMatrix.resultMatrix[idx.y * 8192 + idx.x] = sum;
    }
)";

bool m_break = false;

void INThandler(int sig) {
  signal(sig, SIG_IGN);
  m_break = true;
}

Workload::Workload(){};

Workload::Workload(int cpu, int gpu, bool random) {
  struct timespec init, begin, end, begin_i, end_i;
  
  /* Total execution occurs in duration x size (sec)*/

  int duration;
  int size = 1;
  
  
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
  double elepsed_t = 0;
  int tmp_c, tmp_g;
  duration = 15; // 12 kernels 1 kernel 3sec
  // clock_gettime(CLOCK_MONOTONIC, &begin_i);
  for(int k=0; k<hist_c.size(); k++){
    tmp_c = 0;
    tmp_g = hist_g[k];
    // tmp_c = 20;
    // tmp_g = 20;

    if(tmp_c<0) tmp_c *= -1;
    if(tmp_g<0) tmp_g *= -1;
    
    if(tmp_c%10 <= 5){
      if(tmp_c > 80) cpu = 6;
      else cpu = tmp_c / 10;
    } 
    else {
      if(tmp_c > 80) cpu = 6;
      else cpu = tmp_c/10 + 1;
    }

    if(tmp_g%10 <= 5){
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
    std::cout << "Creates " << gpu * 10 << "% workload gpu worker"
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
    // printf("%.6fs\n", elepsed_t);
    std::cout << "Timeout" << "\n";
    stop = true;
    ignition = false;
    for (auto& workers : gpu_workload_pool) workers.join();
    for (auto& workers : cpu_workload_pool) workers.join();
    cpu_workload_pool.clear();
    gpu_workload_pool.clear();
    std::cout << "=====================\n";
  }
#endif
};

void Workload::CPU_Worker() {
  // not implemented
  std::cout << "Created new CPU worker \n";
  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }
  double a = 1;
  double b = 0.0003;
  while (!stop) {
    a *= b;
  }
  std::cout << "Terminates CPU worker " << "\n";
};


void Workload::GPU_Worker() {
  EGLDisplay display;
  EGLContext context;
  EGLSurface surface;
  int count=1, idx;
  double response_t = 0;
  double tot_response_t = 0;
  struct timespec begin, end;

  // Initialize EGL
  display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  // if (display == EGL_NO_DISPLAY) {
  //   printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
  //   return;
  // }
  EGLBoolean returnValue = eglInitialize(display, NULL, NULL);
  // if (returnValue != EGL_TRUE) {
  //   printf("eglInitialize failed\n");
  //   return;
  // }
  // Configure EGL attributes
  EGLConfig config;
  EGLint numConfigs;
  EGLint configAttribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
  eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);

  // Create an EGL context
  EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};

  context = eglCreateContext(display, EGL_NO_CONTEXT, EGL_CAST(EGLConfig, 0),
                             contextAttribs);
  // if (context == EGL_NO_CONTEXT) {
  //   printf("eglCreateContext failed\n");
  //   return;
  // }
  // Create a surface
  surface = eglCreatePbufferSurface(display, config, NULL);

  // Make the context current
  eglMakeCurrent(display, surface, surface, context);
  // if (returnValue != EGL_TRUE) {
  //   printf("eglMakeCurrent failed returned %d\n", returnValue);
  //   return;
  // }
  // Compile compute shader
  GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(computeShader, 1, &computeShaderSource, NULL);
  glCompileShader(computeShader);

  // Create program and attach shader
  GLuint program = glCreateProgram();
  glAttachShader(program, computeShader);
  glLinkProgram(program);
  GLint linkStatus = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
  // if (!linkStatus) {
  //   printf("glGetProgramiv failed returned \n");
  //   return;
  // }

  // Initialize data
  const long long int matrixElements = GPU_MAT_SIZE * GPU_MAT_SIZE;
  std::vector<float> matrixA(matrixElements);
  std::vector<float> matrixB(matrixElements);
  std::vector<float> resultMatrix(matrixElements);

  for (int i = 0; i < matrixElements; ++i) {
    matrixA[i] = static_cast<float>(i);
    matrixB[i] = static_cast<float>(i + matrixElements);
  }

  // Create buffer objects
  GLuint bufferA, bufferB, bufferResult;
  glGenBuffers(1, &bufferA);
  glGenBuffers(1, &bufferB);
  glGenBuffers(1, &bufferResult);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferA);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferB);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferResult);

  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * matrixElements,
               matrixA.data(), GL_STATIC_DRAW);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * matrixElements,
               matrixB.data(), GL_STATIC_DRAW);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * matrixElements, NULL,
               GL_STATIC_DRAW);

  // Bind buffer objects to binding points
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferA);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferB);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufferResult);

  glUseProgram(program);
  std::cout << "Created new GPU worker \n";

  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }

  signal(SIGINT, INThandler);
  std::ofstream outfile;

  while (!stop) {
    if (m_break) break;
    int PERIOD = 5;
    // glDispatchCompute(16, 16, 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(PERIOD));
    
    clock_gettime(CLOCK_MONOTONIC, &begin);
    glDispatchCompute(16, 1, 1);
    glFlush();  // Ensures that the dispatch command is processed, delete
    // // Create a fence sync object and wait for the GPU to finish
    GLsync syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0); //delete
    glWaitSync(syncObj, 0, GL_TIMEOUT_IGNORED); // delete 
    clock_gettime(CLOCK_MONOTONIC, &end);

    response_t = (end.tv_sec - begin.tv_sec) +
                 ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    tot_response_t += response_t;
    count++;

    //glDeleteSync(syncObj);  // Clean up the sync object
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glFinish();  // all commmand push to GPU HW queue (gpu has two queue, gpu
                 // drvier queue + gpu hw queue )
    printf("%d's elapsed : %.11f\n", count, response_t);
  }
  // printf("%d's average : %.11f\n", count, (tot_response_t / double(count)));

  // Read back result
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferResult);
  float* output = (float*)(glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                            sizeof(float) * matrixElements,
                                            GL_MAP_READ_BIT));

  // Clean up
  glDeleteShader(computeShader);
  glDeleteProgram(program);
  glDeleteBuffers(1, &bufferA);
  glDeleteBuffers(1, &bufferB);
  glDeleteBuffers(1, &bufferResult);

  // Tear down EGL
  eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroySurface(display, surface);
  eglDestroyContext(display, context);
  eglTerminate(display);

  std::cout << "Terminates GPU worker "
            << "\n";
  return;
}

Workload::~Workload(){};
