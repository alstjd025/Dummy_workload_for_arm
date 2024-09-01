#include "dummy_workload.h"

#define GPU_MAT_SIZE 65536
//#define GPU_MAT_SIZE 32768
// 1048576, 524288, 262144, 131072, 65536
// 1048576 - 2.2s, 100% 4.68s
// 524288 - 1s  , 100%  2.37
// 262144 - 0.5s, 100%  1.16s
// 131072 - 0.24s, 100% 0.56
// 65536 - 0.12s, 100%  0.29s
//

const char* computeShaderSource = R"(
    #version 310 es

    layout (local_size_x = 16, local_size_y = 16) in;

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
        for (int k = 0; k < 50000; ++k) {
            sum += inputMatrixA.matrixA[idx.y * 1024 + k] * inputMatrixB.matrixB[k * 1024 + idx.x];
        }
        outputMatrix.resultMatrix[idx.y * 1024 + idx.x] = sum;
    }
)";

Workload::Workload(){};

Workload::Workload(int duration, int cpu, int gpu, bool random) {
  struct timespec begin, end;
  std::cout << "Got cpu " << cpu << " gpu " << gpu << " duration " << duration
            << "\n";
  stop = false;
  if (gpu > 0) {
    gpu_workload_pool.reserve(gpu);
    for (int i = 0; i < gpu; ++i) {
      std::cout << "Creates " << i << " gpu worker"
                << "\n";
      gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
    }
  }
  if (cpu > 0) {
    cpu_workload_pool.reserve(cpu);
    for (int i = 0; i < cpu; ++i) {
      std::cout << "Creates " << i << " cpu worker"
                << "\n";
      cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
    }
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));
  {  // wakes gpu workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers"
              << "\n";
  }

  clock_gettime(CLOCK_MONOTONIC, &begin);
  double elepsed_t = 0;
  while (elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  // std::cout << "Timeout"
  //           << "\n";
  stop = true;
  for (auto& workers : gpu_workload_pool) workers.join();
  for (auto& workers : cpu_workload_pool) workers.join();
  // std::cout << "Dummy workload end"
  //           << "\n";
};

void Workload::CPU_Worker() {
  // not implemented
  struct timespec begin, end;
  double response_t, tot_response_t;
  int count = 0;
  // std::cout << "Created new CPU worker \n";
  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }
  double a = 1;
  double b = 0.0003;
  float arr1[16][5] = {{0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09},
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09},
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09}, 
                      {0.03, 0.02, 0.01, 0.06, 0.09},
                      {0.03, 0.02, 0.01, 0.06, 0.09},
                      };
  float arr2[5][16] = {{0.03, 0.02, 0.01, 0.06, 0.09, 0.03, 0.02, 0.01, 0.06, 0.09,0.03, 0.02, 0.01, 0.06, 0.09, 0.07},
                      {0.03, 0.02, 0.01, 0.06, 0.09, 0.03, 0.02, 0.01, 0.06, 0.09,0.03, 0.02, 0.01, 0.06, 0.09, 0.07},
                      {0.03, 0.02, 0.01, 0.06, 0.09, 0.03, 0.02, 0.01, 0.06, 0.09,0.03, 0.02, 0.01, 0.06, 0.09, 0.07},
                      {0.03, 0.02, 0.01, 0.06, 0.09, 0.03, 0.02, 0.01, 0.06, 0.09,0.03, 0.02, 0.01, 0.06, 0.09, 0.07},
                      {0.03, 0.02, 0.01, 0.06, 0.09, 0.03, 0.02, 0.01, 0.06, 0.09,0.03, 0.02, 0.01, 0.06, 0.09, 0.07}
                      };
  float result[16][16] = {{0,},};

  while (!stop) {
    count++;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for(int i=0;i<16;++i){
      for(int j=0;j<16;++j){
        float sum = 0.0;
        for(int k=0;k<5;++k) sum += arr1[i][k] * arr2[k][j];
        result[i][j] = sum;
      }
    }
    // a *= b;
    clock_gettime(CLOCK_MONOTONIC, &end);
    response_t = (end.tv_sec - begin.tv_sec) +
                 ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    // printf("%.11f\n", count, response_t);
    tot_response_t += response_t;
  }
  //printf("\033[0;31m%.11f\033[0m\n", tot_response_t/count);
  // std::cout << "Terminates CPU worker "
  //           << "\n";
};

void Workload::GPU_Worker() {
  EGLDisplay display;
  EGLContext context;
  EGLSurface surface;

  double response_t = 0;
  double tot_response_t = 0;
  int count = 0;
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

  while (!stop) {
    // glDispatchCompute(16, 16, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    clock_gettime(CLOCK_MONOTONIC, &begin);
    glDispatchCompute(16, 1, 1);
    //glFlush();  // Ensures that the dispatch command is processed
    // Create a fence sync object and wait for the GPU to finish
    //GLsync syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    //glWaitSync(syncObj, 0, GL_TIMEOUT_IGNORED);
    clock_gettime(CLOCK_MONOTONIC, &end);
    response_t = (end.tv_sec - begin.tv_sec) +
                 ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    tot_response_t += response_t;
    count++;
    // glDeleteSync(syncObj);  // Clean up the sync object
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glFinish();  // all commmand push to GPU HW queue (gpu has two queue, gpu
                 // drvier queue + gpu hw queue )
    //printf("%d's elapsed : %.11f\n", count, response_t);
  }
  //printf("%d's average : %.11f\n", count, (tot_response_t / double(count)));

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
