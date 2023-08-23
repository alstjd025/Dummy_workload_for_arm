#include "dummy_workload.h"

#define GPU_MAT_SIZE 131072
// 1048576, 524288, 262144, 131072, 65536
// 1048576 - 2.2s, 100%
// 524288 - 1s  , 100%
// 262144 - 0.5s, 100%
// 131072 - 0.24s, 100%
// 65536 - 0.12s, 100%

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
        for (int k = 0; k < 131072; ++k) {
            sum += inputMatrixA.matrixA[idx.y * 131072 + k] * inputMatrixB.matrixB[k * 131072 + idx.x];
        }
        outputMatrix.resultMatrix[idx.y * 131072 + idx.x] = sum;
    }
)";

Workload::Workload() {};

Workload::Workload(int duration, int cpu, int gpu, bool random){
  struct timespec begin, end;
  std::cout << "Got cpu " << cpu << " gpu " << gpu << " duration " << duration << "\n";
  stop = false;
  if(gpu > 0){
    gpu_workload_pool.reserve(gpu);
    for(int i=0; i<gpu; ++i){
      std::cout << "Creates " << i << " gpu worker" << "\n";
      gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
    }
  }
  if(cpu > 0){
    cpu_workload_pool.reserve(cpu);
    for(int i=0; i<cpu; ++i){
      std::cout << "Creates " << i << " cpu worker" << "\n";
      cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
    }
  }
  
  std::this_thread::sleep_for(std::chrono::seconds(1));
  { // wakes gpu workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers" << "\n";
  }

  clock_gettime(CLOCK_MONOTONIC, &begin);
  double elepsed_t = 0;
  while(elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  std::cout << "Timeout" << "\n";
  stop = true;
  for(auto& workers : gpu_workload_pool)
    workers.join();
  for(auto& workers : cpu_workload_pool)
    workers.join();
  std::cout << "Dummy workload end" << "\n";
};

void Workload::CPU_Worker(){
  // not implemented
  std::cout << "Created new CPU worker \n";
  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }
  double a = 1;
  double b = 0.0003;
  while(!stop){
    a *= b;
  }
  std::cout << "Terminates CPU worker " << "\n";
};

void Workload::GPU_Worker(){
  EGLDisplay display;
  EGLContext context;
  EGLSurface surface;

  // Initialize EGL
  display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (display == EGL_NO_DISPLAY) {
      printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
      return;
  }
  EGLBoolean returnValue = eglInitialize(display, NULL, NULL);
  if (returnValue != EGL_TRUE) {
      printf("eglInitialize failed\n");
      return;
  }
  // Configure EGL attributes
  EGLConfig config;
  EGLint numConfigs;
  EGLint configAttribs[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_NONE
  };
  eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);

  // Create an EGL context
  EGLint contextAttribs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3,
      EGL_NONE
  };
  
  context = eglCreateContext(display, EGL_NO_CONTEXT, EGL_CAST(EGLConfig,0), contextAttribs);
  if (context == EGL_NO_CONTEXT) {
      printf("eglCreateContext failed\n");
      return;
  }
  // Create a surface
  surface = eglCreatePbufferSurface(display, config, NULL);

  // Make the context current
  returnValue = eglMakeCurrent(display, surface, surface, context);
  if (returnValue != EGL_TRUE) {
      printf("eglMakeCurrent failed returned %d\n", returnValue);
      return;
  }
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
  if (!linkStatus){
      printf("glGetProgramiv failed returned \n");
      return;
  }


  // Initialize data
  const int matrixElements = GPU_MAT_SIZE * GPU_MAT_SIZE;
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

  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * matrixElements, matrixA.data(), GL_STATIC_DRAW);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * matrixElements, matrixB.data(), GL_STATIC_DRAW);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * matrixElements, NULL, GL_STATIC_DRAW);

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
  while(!stop){
    glDispatchCompute(16, 16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glFinish();
  }

  // Read back result
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferResult);
  float *output = (float *)(glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 
      0, sizeof(float) * matrixElements, GL_MAP_READ_BIT));

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

  std::cout << "Terminates GPU worker " << "\n";
  return;
}

Workload::~Workload() {};