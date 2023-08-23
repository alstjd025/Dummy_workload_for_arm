#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <GLES3/gl31.h>
#include <EGL/egl.h>

class Workload{
  public:
    Workload();  
    Workload(int duration, int cpu, int gpu, bool random);
    
    ~Workload();

    void GPU_Worker(int& id, std::atomic_bool& stop_flag);
    void CPU_Worker(int id);

    struct timespec start_time;
    bool ignition = false;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic_bool stop;
  private:
    // GPU workload pool 
    std::vector<std::thread> gpu_workload_pool;

    // CPU workload pool 
    std::vector<std::thread> cpu_workload_pool;



}; // class Workload