#include <GLES3/gl31.h>
#include <EGL/egl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

const int matrixSize = 262144;

// 1048576 - 250%, 524288 - 100%, 262144, 131072
// 
// 
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
        for (int k = 0; k < 262144; ++k) {
            sum += inputMatrixA.matrixA[idx.y * 262144 + k] * inputMatrixB.matrixB[k * 262144 + idx.x];
        }
        outputMatrix.resultMatrix[idx.y * 262144 + idx.x] = sum;
    }
)";

int main() {
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;

    // Initialize EGL
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
        return 0;
    }
    EGLBoolean returnValue = eglInitialize(display, NULL, NULL);
    if (returnValue != EGL_TRUE) {
        printf("eglInitialize failed\n");
        return 0;
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
        return 0;
    }
    // Create a surface
    surface = eglCreatePbufferSurface(display, config, NULL);

    // Make the context current
    returnValue = eglMakeCurrent(display, surface, surface, context);
    if (returnValue != EGL_TRUE) {
        printf("eglMakeCurrent failed returned %d\n", returnValue);
        return 0;
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
        return 0;
    }


    // Initialize data
    const int matrixElements = matrixSize * matrixSize;
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

    // NO IDEA ON CHOOSING WORKGROUP SIZE...
    // // Dispatch compute shader with appropriate workgroups
    // GLuint numWorkgroupsX = (matrixSize + 31) / 32; // Divide and round up
    // GLuint numWorkgroupsY = (matrixSize + 31) / 32; // Divide and round up
    
    glUseProgram(program);

    // glDispatchCompute(numWorkgroupsX, numWorkgroupsY, 1);
    
    glDispatchCompute(16, 16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    // Read back result
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferResult);
    float *output = (float *)(glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 
        0, sizeof(float) * matrixElements, GL_MAP_READ_BIT));

    // Print result
    // std::cout << "Result Matrix:" << std::endl;
    // // std::cout << output[0] << " ";
    // for (GLuint i = 0; i < matrixSize; ++i) {
    //     for (GLuint j = 0; j < matrixSize; ++j) {
    //         std::cout << output[i * matrixSize + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "MAX WORKGROUP size " << GL_MAX_COMPUTE_WORK_GROUP_SIZE << "\n" ;
    std::cout << "MAX WORKGROUP count " << GL_MAX_COMPUTE_WORK_GROUP_COUNT << "\n" ;
    std::cout << "MAX WORKGROUP invoke " << GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS << "\n" ;

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

    return 0;
}
