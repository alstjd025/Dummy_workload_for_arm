#include "dummy_workload.h"

int main(int argv, char* argc[]){
  if(argv < 2){
    std::cout << "Not enough args, usage : duration(0~)," <<
                 "num_cpu(0~), num_gpu(0~), random(0, 1)" << "\n";
    exit(-1);
  }
  int cpu, gpu, duration;
  duration = atoi(argc[1]);
  cpu = atoi(argc[2]);
  gpu = atoi(argc[3]);
  
  Workload workload(duration, cpu, gpu, false);

  return 0;
}