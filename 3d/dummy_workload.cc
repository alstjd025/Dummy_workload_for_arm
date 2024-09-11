#include "dummy_workload.h"
#define CPU_UTIL_FILE "/home/nvidia/TfLite_apps/scheduler/cpu_util"
#define GPU_UTIL_FILE "/home/nvidia/TfLite_apps/scheduler/gpu_util"
#define GPU_MAT_SIZE 8192 // 655356(25000, PERIOD=30 -> aoubt 22%) 65536(50000, PERIOD=15 -> about 40%) 65536(66535, PERIOD=7 -> about 60%) 65536(66535, PERIOD=2 -> about 80%)
#define need_period
#define cpu_use
#define gpu_use
#define max_cpu 6
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

Workload::Workload(int cpu, int gpu, bool random) {
  struct timespec init, begin, end, begin_i, end_i;
  
  /* Total execution occurs in duration x size (sec)*/

  double duration, total_duration;
  int size = 1;

  clock_gettime(CLOCK_MONOTONIC, &init);
  std::ofstream gpu_util_f, cpu_util_f;
  ////////////// Init 0 0
  std::cout << "========Init=========\n";
  ///////////////////////////////////////////////////////////////////////
  ////// workload start 
  double elepsed_t = 0;
  double total_elepsed_t = 0;
  int tmp_c, tmp_g;
  duration = 5; // 12 kernels 1 kernel 3sec
  total_duration = 15;

    gpu_workload_pool.reserve(1);
    std::cout << "Creates " << gpu * 10 << "% workload gpu worker"
              << "\n";
    gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
    
  
    cpu_worker_termination = false;
    gpu_worker_termination = false;
    clock_gettime(CLOCK_MONOTONIC, &init);
    while (total_elepsed_t < total_duration) {
      ////////////////////////
      // GPU start (300ms)  //
      ////////////////////////
      gpu_stop = false;
      {  // wakes  workers
        std::unique_lock<std::mutex> lock(gpu_mtx);
        gpu_ignition = true;
        gpu_cv.notify_all();
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
      gpu_stop = true;
      gpu_ignition = false;
      total_elepsed_t += elepsed_t;
      // printf("%.6fs\n", elepsed_t);
      std::cout << "GPU workload done" << "\n";
      printf("total eplepsed t : %f \n", total_elepsed_t);
    }
    gpu_worker_termination = true;

    for (auto& workers : gpu_workload_pool) workers.join();
    for (auto& workers : cpu_workload_pool) workers.join();
    cpu_workload_pool.clear();
    gpu_workload_pool.clear();
    std::cout << "=====================\n";
  }

void Workload::CPU_Worker() {
  // not implemented
  while(!cpu_worker_termination){
    std::cout << "Created new CPU worker \n";
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


  int x1 = 1024, y1 = 1024, z1 = 128; // Matrix A size (4x4x4)
  int x2 = 1024, y2 = 1024, z2 = 8; // Matrix B size (4x4x4)

  // Initialize matrices A and B with some data
  std::vector<float> A(x1 * y1 * z1, 1.0f); // Fill with 1.0f for simplicity
  std::vector<float> B(x2 * y2 * z2, 2.0f); // Fill with 2.0f for simplicity
  std::vector<float> C(x1 * y2 * z2, 0.0f); // Result matrix initialized to 0.0f

  // Create buffer objects
  GLuint bufferA, bufferB, bufferC;
  glGenBuffers(1, &bufferA);
  glGenBuffers(1, &bufferB);
  glGenBuffers(1, &bufferC);


  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferA);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferB);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferC);


  glBufferData(GL_SHADER_STORAGE_BUFFER, A.size() * sizeof(float),
               A.data(), GL_STATIC_DRAW);
  glBufferData(GL_SHADER_STORAGE_BUFFER, B.size() * sizeof(float),
               B.data(), GL_STATIC_DRAW);
  glBufferData(GL_SHADER_STORAGE_BUFFER, C.size() * sizeof(float),
               C.data(), GL_STATIC_DRAW);
              
  

  // Bind buffer objects to binding points
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferA);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferB);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufferC);
  
  glUniform3i(glGetUniformLocation(program, "sizeA"), x1, y1, z1);
  glUniform3i(glGetUniformLocation(program, "sizeB"), x2, y2, z2);

  glUseProgram(program);
  std::cout << "Created new GPU worker \n";
  while(!gpu_worker_termination){
    {
      std::unique_lock<std::mutex> lock_(gpu_mtx);
      gpu_cv.wait(lock_, [this]() { return gpu_ignition; });
    }

    signal(SIGINT, INThandler);
    std::ofstream outfile;

    // multi-level test
    // Todo : 
    while (!gpu_stop) {
      if (m_break) break;
      // std::this_thread::sleep_for(std::chrono::milliseconds(PERIOD));
      
      clock_gettime(CLOCK_MONOTONIC, &begin);
      // glDispatchCompute(16, 16, 16);
      glDispatchCompute((GLuint)x1, (GLuint)y2, (GLuint)z2);
      glFlush();  // Ensures that the dispatch command is processed, delete
      // // Create a fence sync object and wait for the GPU to finish
      GLsync syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0); //delete
      glWaitSync(syncObj, 0, GL_TIMEOUT_IGNORED); // delete 
      clock_gettime(CLOCK_MONOTONIC, &end);

      response_t = (end.tv_sec - begin.tv_sec) +
                   ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      tot_response_t += response_t;
      count++;

      glDeleteSync(syncObj);  // Clean up the sync object
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      glFinish();  // all commmand push to GPU HW queue (gpu has two queue, gpu
                  // drvier queue + gpu hw queue )
      printf("%d's elapsed : %.11f\n", count, response_t);
    }
  printf("%d's average : %.11f\n", count, (tot_response_t / double(count)));
  }

  // Read back result
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferC);
  float* output = (float*)(glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                            sizeof(float) * C.size(),
                                            GL_MAP_READ_BIT));
  // std::cout << "Result matrix C:" << "\n";
  // for (int i = 0; i < x1; ++i) {
  //     for (int j = 0; j < y2; ++j) {
  //         for (int k = 0; k < z2; ++k) {
  //             std::cout << C[i * (y2 * z2) + j * z2 + k] << " ";
  //         }
  //         std::cout << "\n";
  //     }
  //     std::cout << "--------" << "\n";
  // }

  // Clean up
  glDeleteShader(computeShader);
  glDeleteProgram(program);
  glDeleteBuffers(1, &bufferA);
  glDeleteBuffers(1, &bufferB);
  glDeleteBuffers(1, &bufferC);

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
