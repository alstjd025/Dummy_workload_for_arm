#include "dummy_workload.h"
#define CPU_UTIL_FILE "/home/nano/TfLite_apps/scheduler/cpu_util"
#define GPU_UTIL_FILE "/home/nano/TfLite_apps/scheduler/gpu_util"
#define GPU_MAT_SIZE 512
// 1048576, 524288, 262144, 131072, 65536
// 1048576 - 2.2s, 100%
// 524288 - 1s  , 100%
// 262144 - 0.5s, 100%
// 131072 - 0.24s, 100%
// 65536 - 0.12s - glDispatchCompute(32, 32, 1); 98%
//
// 1024 - glDispatchCompute(32, 32, 1); 98%
// 1024 - glDispatchCompute(1, 1, 1); 50%

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
        for (int k = 0; k < 1024; ++k) {
            sum += inputMatrixA.matrixA[idx.y * 1024 + k] * inputMatrixB.matrixB[k * 1024 + idx.x];
        }
        outputMatrix.resultMatrix[idx.y * 1024 + idx.x] = sum;
    }
)";

Workload::Workload() {};

Workload::Workload(int duration, int cpu, int gpu, bool random){
  struct timespec init, begin, end;
  clock_gettime(CLOCK_MONOTONIC, &init);
  std::ofstream gpu_util_f, cpu_util_f;
  std::cout << "Got cpu " << cpu << " gpu " << gpu << " duration " << duration << "\n";
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
  if(cpu > 0){
    cpu_workload_pool.reserve(cpu);
    for(int i=0; i<cpu; ++i){
      // std::cout << "Creates " << i << " cpu worker" << "\n";
      cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
    }
  }
  
  std::this_thread::sleep_for(std::chrono::seconds(1));
  { // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers" << "\n";
  }
  
  double elepsed_t = 0;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  elepsed_t = (begin.tv_sec - init.tv_sec) + ((begin.tv_nsec - init.tv_nsec) / 1000000000.0);
  printf("start time : %.6f \n");
  while(elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  // std::cout << "Timeout" << "\n";
  stop = true;
  ignition = false;
  for(auto& workers : gpu_workload_pool)
    workers.join();
  for(auto& workers : cpu_workload_pool)
    workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();

  ///////// CPU 400 GPU 0
  cpu = 4;
  duration = 10;
  cpu_workload_pool.reserve(cpu);
  stop = false;
  for(int i=0; i<cpu; ++i){
    // std::cout << "Creates " << i << " cpu worker" << "\n";
    cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
  }
  cpu_util_f << "400" << "\n";
  cpu_util_f.close();
  gpu_util_f << "0" << "\n";
  gpu_util_f.close();


  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  { // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers" << "\n";
  }
  clock_gettime(CLOCK_MONOTONIC, &begin);
  elepsed_t = 0;
  while(elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  std::cout << "Timeout" << "\n";
  stop = true;
  ignition = false;
  for(auto& workers : gpu_workload_pool)
    workers.join();
  for(auto& workers : cpu_workload_pool)
    workers.join();
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
  for(int i=0; i<1; ++i){
    std::cout << "Creates " << i << " gpu worker" << "\n";
    gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  }
  cpu_util_f << "0" << "\n";
  cpu_util_f.close();
  gpu_util_f << "100" << "\n";
  gpu_util_f.close();
  
  stop = false;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  { // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers" << "\n";
  }
  clock_gettime(CLOCK_MONOTONIC, &begin);
  elepsed_t = 0;
  while(elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  std::cout << "Timeout" << "\n";
  stop = true;
  ignition = false;
  for(auto& workers : gpu_workload_pool)
    workers.join();
  for(auto& workers : cpu_workload_pool)
    workers.join();
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
  for(int i=0; i<cpu; ++i){
    std::cout << "Creates " << i << " cpu worker" << "\n";
    cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
  }
  gpu_workload_pool.reserve(1);
  for(int i=0; i<1; ++i){
    std::cout << "Creates " << i << " gpu worker" << "\n";
    gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  }
  cpu_util_f << "200" << "\n";
  cpu_util_f.close();
  gpu_util_f << "100" << "\n";
  gpu_util_f.close();

  stop = false;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  { // wakes  workers
    std::unique_lock<std::mutex> lock(mtx);
    ignition = true;
    cv.notify_all();
    std::cout << "Notified all workers" << "\n";
  }
  clock_gettime(CLOCK_MONOTONIC, &begin);
  elepsed_t = 0;
  while(elepsed_t < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    clock_gettime(CLOCK_MONOTONIC, &end);
    elepsed_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  }
  std::cout << "Timeout" << "\n";
  stop = true;
  ignition = false;
  for(auto& workers : gpu_workload_pool)
    workers.join();
  for(auto& workers : cpu_workload_pool)
    workers.join();
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
  cpu_util_f << "0" << "\n";
  cpu_util_f.close();
  gpu_util_f << "0" << "\n";
  gpu_util_f.close();
  std::cout << "Dummy workload end" << "\n";
};

void Workload::CPU_Worker(){
  // not implemented
  // std::cout << "Created new CPU worker \n";
  {
    std::unique_lock<std::mutex> lock_(mtx);
    cv.wait(lock_, [this]() { return ignition; });
  }
  double a = 1;
  double b = 0.0003;
  while(!stop){
    a *= b;
  }
  // std::cout << "Terminates CPU worker " << "\n";
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
    // struct timespec begin, end;
    // clock_gettime(CLOCK_MONOTONIC, &begin);
    // glDispatchCompute(16, 16, 1);
    glDispatchCompute(32, 32, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glFinish();
    // clock_gettime(CLOCK_MONOTONIC, &end);
    // double latency = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    // std::cout << "latency " << latency << "\n";
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
