/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "gl_sync.h"

bool GlSyncWait() {
  GlSync sync;
  GlSync::NewSync(&sync);
  // Flush sync and loop afterwards without it.
  GLenum status = glClientWaitSync(sync.sync(), GL_SYNC_FLUSH_COMMANDS_BIT,
                                   /* timeout ns = */ 0);
  while (true) {
    switch (status) {
      case GL_TIMEOUT_EXPIRED:
        break;
      case GL_CONDITION_SATISFIED:
      case GL_ALREADY_SIGNALED:
        return true;
      case GL_WAIT_FAILED:
        return false;
    }
    status = glClientWaitSync(sync.sync(), 0, /* timeout ns = */ 10000000);
  }
  return true;
}

bool GlActiveSyncWait() {
  GlSync sync;
  GlSync::NewSync(&sync);
  // Since creating a Sync object is itself a GL command it *must* be flushed.
  // Otherwise glGetSynciv may never succeed. Perform a flush with
  // glClientWaitSync call.
  GLenum status = glClientWaitSync(sync.sync(), GL_SYNC_FLUSH_COMMANDS_BIT,
                                   /* timeout ns = */ 0);
  switch (status) {
    case GL_TIMEOUT_EXPIRED:
      break;
    case GL_CONDITION_SATISFIED:
    case GL_ALREADY_SIGNALED:
      return true;
    case GL_WAIT_FAILED:
      return false;
  }

  // Start active loop.
  GLint result = GL_UNSIGNALED;
  while (true) {
    glGetSynciv(sync.sync(), GL_SYNC_STATUS, sizeof(GLint), nullptr, &result);
    if (result == GL_SIGNALED) {
      return true;
    }
#ifdef __ARM_ACLE
    // Try to save CPU power by yielding CPU to another thread.
    __yield();
#endif
  }
}

bool GlShaderSync::NewSync(GlShaderSync* gl_sync) {
  GlShaderSync sync;
  CreatePersistentBuffer(sizeof(int), &sync.flag_buffer_);
  static const std::string* kCode = new std::string(R"(#version 310 es
  layout(local_size_x = 1, local_size_y = 1) in;
  layout(std430) buffer;
  layout(binding = 0) buffer Output {
    int elements[];
  } output_data;
  void main() {
    output_data.elements[0] = 1;
  })");
  GlShader shader;
  GlShader::CompileShader(GL_COMPUTE_SHADER, *kCode, &shader);
  GlProgram::CreateWithShader(shader, &sync.flag_program_);
  *gl_sync = std::move(sync);
  return true;
}

// How it works: GPU writes a buffer and CPU checks the buffer value to be
// changed. The buffer is accessible for writing by GPU and reading by CPU
// simultaneously - persistent buffer or buffer across shild context can be used
// for that.
bool GlShaderSync::Wait() {
  if (!flag_buffer_.is_valid()) {
    std::cout << "GlShaderSync is not initialized." << "\n";
    return false;
  }
  flag_buffer_.BindToIndex(0);
  volatile int* flag_ptr_ = reinterpret_cast<int*>(flag_buffer_.data());
  *flag_ptr_ = 0;
  flag_program_.Dispatch({1, 1, 1});
  // glFlush must be called to upload GPU task. Adreno won't start executing
  // the task without glFlush.
  glFlush();
  // Wait for the value is being updated by the shader.
  while (*flag_ptr_ != 1) {
  }
  return true;
}

