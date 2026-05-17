/*
 * OpenCL Dynamic Shim — Q-NarwhalKnight Miner
 *
 * This shim replaces the hard libOpenCL.so link dependency.
 * At runtime it tries to dlopen the real libOpenCL; if absent
 * it returns error codes so the miner falls back to CPU mining.
 *
 * Auto-generated from opencl-sys-0.2.9/src/cl.rs
 */
#ifdef __linux__
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

static void* g_opencl_lib = ((void*)0);
static pthread_once_t g_opencl_once = PTHREAD_ONCE_INIT;

static void opencl_shim_init(void) {
    g_opencl_lib = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_opencl_lib)
        g_opencl_lib = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_GLOBAL);
}

static void* get_sym(const char* sym) {
    pthread_once(&g_opencl_once, opencl_shim_init);
    if (!g_opencl_lib) return ((void*)0);
    return dlsym(g_opencl_lib, sym);
}

int32_t clGetPlatformIDs(uint32_t num_entries, void* platforms, void* num_platforms) {
    typedef int32_t (*fn_t)(uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetPlatformIDs");
    if (!fn) return -1001;
    return fn(num_entries, platforms, num_platforms);
}

int32_t clGetPlatformInfo(void* platform, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetPlatformInfo");
    if (!fn) return -1001;
    return fn(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clGetDeviceIDs(void* platform, uint64_t device_type, uint32_t num_entries, void* devices, void* num_devices) {
    typedef int32_t (*fn_t)(void*, uint64_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetDeviceIDs");
    if (!fn) return -1001;
    return fn(platform, device_type, num_entries, devices, num_devices);
}

int32_t clGetDeviceInfo(void* device, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetDeviceInfo");
    if (!fn) return -1001;
    return fn(device, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clCreateSubDevices(void* in_device, void* properties, uint32_t num_devices, void* out_devices, void* num_devices_ret) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateSubDevices");
    if (!fn) return -1001;
    return fn(in_device, properties, num_devices, out_devices, num_devices_ret);
}

int32_t clRetainDevice(void* device) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clRetainDevice");
    if (!fn) return -1001;
    return fn(device);
}

int32_t clReleaseDevice(void* device) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clReleaseDevice");
    if (!fn) return -1001;
    return fn(device);
}

int32_t clSetDefaultDeviceCommandQueue(void* context, void* device, void* command_queue) {
    typedef int32_t (*fn_t)(void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clSetDefaultDeviceCommandQueue");
    if (!fn) return -1001;
    return fn(context, device, command_queue);
}

int32_t clGetDeviceAndHostTimer(void* device, void* device_timestamp, void* host_timestamp) {
    typedef int32_t (*fn_t)(void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetDeviceAndHostTimer");
    if (!fn) return -1001;
    return fn(device, device_timestamp, host_timestamp);
}

int32_t clGetHostTimer(void* device, void* host_timestamp) {
    typedef int32_t (*fn_t)(void*, void*);
    fn_t fn = (fn_t)get_sym("clGetHostTimer");
    if (!fn) return -1001;
    return fn(device, host_timestamp);
}

void* clCreateContext(void* properties, uint32_t num_devices, void* devices, void* pfn_notify, void* user_data, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint32_t, void*, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateContext");
    if (!fn) return NULL;
    return fn(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

void* clCreateContextFromType(void* properties, uint64_t device_type, void* pfn_notify, void* user_data, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint64_t, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateContextFromType");
    if (!fn) return NULL;
    return fn(properties, device_type, pfn_notify, user_data, errcode_ret);
}

int32_t clRetainContext(void* context) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clRetainContext");
    if (!fn) return -1001;
    return fn(context);
}

int32_t clReleaseContext(void* context) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clReleaseContext");
    if (!fn) return -1001;
    return fn(context);
}

int32_t clGetContextInfo(void* context, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetContextInfo");
    if (!fn) return -1001;
    return fn(context, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clSetContextDestructorCallback(void* context, void* pfn_notify, void* user_data) {
    typedef int32_t (*fn_t)(void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clSetContextDestructorCallback");
    if (!fn) return -1001;
    return fn(context, pfn_notify, user_data);
}

void* clCreateCommandQueueWithProperties(void* context, void* device, void* properties, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateCommandQueueWithProperties");
    if (!fn) return NULL;
    return fn(context, device, properties, errcode_ret);
}

int32_t clRetainCommandQueue(void* command_queue) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clRetainCommandQueue");
    if (!fn) return -1001;
    return fn(command_queue);
}

int32_t clReleaseCommandQueue(void* command_queue) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clReleaseCommandQueue");
    if (!fn) return -1001;
    return fn(command_queue);
}

int32_t clGetCommandQueueInfo(void* command_queue, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetCommandQueueInfo");
    if (!fn) return -1001;
    return fn(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
}

void* clCreateBuffer(void* context, uint64_t flags, size_t size, void* host_ptr, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint64_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateBuffer");
    if (!fn) return NULL;
    return fn(context, flags, size, host_ptr, errcode_ret);
}

void* clCreateSubBuffer(void* buffer, uint64_t flags, int32_t buffer_create_type, void* buffer_create_info, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint64_t, int32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateSubBuffer");
    if (!fn) return NULL;
    return fn(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);
}

void* clCreateImage(void* context, uint64_t flags, void* image_format, void* image_desc, void* host_ptr, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint64_t, void*, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateImage");
    if (!fn) return NULL;
    return fn(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

void* clCreatePipe(void* context, uint64_t flags, uint32_t pipe_packet_size, uint32_t pipe_max_packets, void* properties, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint64_t, uint32_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreatePipe");
    if (!fn) return NULL;
    return fn(context, flags, pipe_packet_size, pipe_max_packets, properties, errcode_ret);
}

void* clCreateBufferWithProperties(void* context, void* properties, uint64_t flags, size_t size, void* host_ptr, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*, uint64_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateBufferWithProperties");
    if (!fn) return NULL;
    return fn(context, properties, flags, size, host_ptr, errcode_ret);
}

void* clCreateImageWithProperties(void* context, void* properties, uint64_t flags, void* image_format, void* image_desc, void* host_ptr, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*, uint64_t, void*, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateImageWithProperties");
    if (!fn) return NULL;
    return fn(context, properties, flags, image_format, image_desc, host_ptr, errcode_ret);
}

int32_t clRetainMemObject(void* memobj) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clRetainMemObject");
    if (!fn) return -1001;
    return fn(memobj);
}

int32_t clReleaseMemObject(void* memobj) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clReleaseMemObject");
    if (!fn) return -1001;
    return fn(memobj);
}

int32_t clGetSupportedImageFormats(void* context, uint64_t flags, int32_t image_type, uint32_t num_entries, void* image_formats, void* num_image_formats) {
    typedef int32_t (*fn_t)(void*, uint64_t, int32_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetSupportedImageFormats");
    if (!fn) return -1001;
    return fn(context, flags, image_type, num_entries, image_formats, num_image_formats);
}

int32_t clGetMemObjectInfo(void* memobj, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetMemObjectInfo");
    if (!fn) return -1001;
    return fn(memobj, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clGetImageInfo(void* image, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetImageInfo");
    if (!fn) return -1001;
    return fn(image, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clGetPipeInfo(void* pipe, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetPipeInfo");
    if (!fn) return -1001;
    return fn(pipe, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clSetMemObjectDestructorCallback(void* memobj, void* pfn_notify, void* user_data) {
    typedef int32_t (*fn_t)(void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clSetMemObjectDestructorCallback");
    if (!fn) return -1001;
    return fn(memobj, pfn_notify, user_data);
}

void* clSVMAlloc(void* context, uint64_t flags, size_t size, uint32_t alignment) {
    typedef void* (*fn_t)(void*, uint64_t, size_t, uint32_t);
    fn_t fn = (fn_t)get_sym("clSVMAlloc");
    if (!fn) return NULL;
    return fn(context, flags, size, alignment);
}

void clSVMFree(void* context, void* svm_pointer) {
    typedef void (*fn_t)(void*, void*);
    fn_t fn = (fn_t)get_sym("clSVMFree");
    if (fn) fn(context, svm_pointer);
}

void* clCreateSamplerWithProperties(void* context, void* normalized_coords, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateSamplerWithProperties");
    if (!fn) return NULL;
    return fn(context, normalized_coords, errcode_ret);
}

int32_t clRetainSampler(void* sampler) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clRetainSampler");
    if (!fn) return -1001;
    return fn(sampler);
}

int32_t clReleaseSampler(void* sampler) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clReleaseSampler");
    if (!fn) return -1001;
    return fn(sampler);
}

int32_t clGetSamplerInfo(void* sampler, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetSamplerInfo");
    if (!fn) return -1001;
    return fn(sampler, param_name, param_value_size, param_value, param_value_size_ret);
}

void* clCreateProgramWithSource(void* context, uint32_t count, const char** strings, void* lengths, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint32_t, const char**, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateProgramWithSource");
    if (!fn) return NULL;
    return fn(context, count, strings, lengths, errcode_ret);
}

void* clCreateProgramWithBinary(void* context, uint32_t num_devices, void* device_list, void* lengths, const unsigned char** binaries, void* binary_status, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint32_t, void*, void*, const unsigned char**, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateProgramWithBinary");
    if (!fn) return NULL;
    return fn(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
}

void* clCreateProgramWithBuiltInKernels(void* context, uint32_t num_devices, void* device_list, char* kernel_names, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint32_t, void*, char*, void*);
    fn_t fn = (fn_t)get_sym("clCreateProgramWithBuiltInKernels");
    if (!fn) return NULL;
    return fn(context, num_devices, device_list, kernel_names, errcode_ret);
}

void* clCreateProgramWithIL(void* context, void* il, size_t length, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*, size_t, void*);
    fn_t fn = (fn_t)get_sym("clCreateProgramWithIL");
    if (!fn) return NULL;
    return fn(context, il, length, errcode_ret);
}

int32_t clRetainProgram(void* program) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clRetainProgram");
    if (!fn) return -1001;
    return fn(program);
}

int32_t clReleaseProgram(void* program) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clReleaseProgram");
    if (!fn) return -1001;
    return fn(program);
}

int32_t clBuildProgram(void* program, uint32_t num_devices, void* device_list, char* options, void* pfn_notify, void* user_data) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*, char*, void*, void*);
    fn_t fn = (fn_t)get_sym("clBuildProgram");
    if (!fn) return -1001;
    return fn(program, num_devices, device_list, options, pfn_notify, user_data);
}

int32_t clCompileProgram(void* program, uint32_t num_devices, void* device_list, char* options, uint32_t num_input_headers, void* input_headers, const char** header_include_names, void* pfn_notify, void* user_data) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*, char*, uint32_t, void*, const char**, void*, void*);
    fn_t fn = (fn_t)get_sym("clCompileProgram");
    if (!fn) return -1001;
    return fn(program, num_devices, device_list, options, num_input_headers, input_headers, header_include_names, pfn_notify, user_data);
}

void* clLinkProgram(void* context, uint32_t num_devices, void* device_list, char* options, uint32_t num_input_programs, void* input_programs, void* pfn_notify, void* user_data, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint32_t, void*, char*, uint32_t, void*, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clLinkProgram");
    if (!fn) return NULL;
    return fn(context, num_devices, device_list, options, num_input_programs, input_programs, pfn_notify, user_data, errcode_ret);
}

int32_t clSetProgramReleaseCallback(void* program, void* pfn_notify, void* user_data) {
    typedef int32_t (*fn_t)(void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clSetProgramReleaseCallback");
    if (!fn) return -1001;
    return fn(program, pfn_notify, user_data);
}

int32_t clSetProgramSpecializationConstant(void* program, uint32_t spec_id, size_t spec_size, void* spec_value) {
    typedef int32_t (*fn_t)(void*, uint32_t, size_t, void*);
    fn_t fn = (fn_t)get_sym("clSetProgramSpecializationConstant");
    if (!fn) return -1001;
    return fn(program, spec_id, spec_size, spec_value);
}

int32_t clUnloadPlatformCompiler(void* platform) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clUnloadPlatformCompiler");
    if (!fn) return -1001;
    return fn(platform);
}

int32_t clGetProgramInfo(void* program, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetProgramInfo");
    if (!fn) return -1001;
    return fn(program, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clGetProgramBuildInfo(void* program, void* device, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetProgramBuildInfo");
    if (!fn) return -1001;
    return fn(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

void* clCreateKernel(void* program, char* kernel_name, void* errcode_ret) {
    typedef void* (*fn_t)(void*, char*, void*);
    fn_t fn = (fn_t)get_sym("clCreateKernel");
    if (!fn) return NULL;
    return fn(program, kernel_name, errcode_ret);
}

int32_t clCreateKernelsInProgram(void* program, uint32_t num_kernels, void* kernels, void* num_kernels_ret) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateKernelsInProgram");
    if (!fn) return -1001;
    return fn(program, num_kernels, kernels, num_kernels_ret);
}

void* clCloneKernel(void* source_kernel, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*);
    fn_t fn = (fn_t)get_sym("clCloneKernel");
    if (!fn) return NULL;
    return fn(source_kernel, errcode_ret);
}

int32_t clRetainKernel(void* kernel) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clRetainKernel");
    if (!fn) return -1001;
    return fn(kernel);
}

int32_t clReleaseKernel(void* kernel) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clReleaseKernel");
    if (!fn) return -1001;
    return fn(kernel);
}

int32_t clSetKernelArg(void* kernel, uint32_t arg_index, size_t arg_size, void* arg_value) {
    typedef int32_t (*fn_t)(void*, uint32_t, size_t, void*);
    fn_t fn = (fn_t)get_sym("clSetKernelArg");
    if (!fn) return -1001;
    return fn(kernel, arg_index, arg_size, arg_value);
}

int32_t clSetKernelArgSVMPointer(void* kernel, uint32_t arg_index, void* arg_value) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*);
    fn_t fn = (fn_t)get_sym("clSetKernelArgSVMPointer");
    if (!fn) return -1001;
    return fn(kernel, arg_index, arg_value);
}

int32_t clSetKernelExecInfo(void* kernel, uint32_t param_name, size_t param_value_size, void* param_value) {
    typedef int32_t (*fn_t)(void*, uint32_t, size_t, void*);
    fn_t fn = (fn_t)get_sym("clSetKernelExecInfo");
    if (!fn) return -1001;
    return fn(kernel, param_name, param_value_size, param_value);
}

int32_t clGetKernelInfo(void* kernel, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetKernelInfo");
    if (!fn) return -1001;
    return fn(kernel, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clGetKernelArgInfo(void* kernel, uint32_t arg_indx, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, uint32_t, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetKernelArgInfo");
    if (!fn) return -1001;
    return fn(kernel, arg_indx, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clGetKernelWorkGroupInfo(void* kernel, void* device, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetKernelWorkGroupInfo");
    if (!fn) return -1001;
    return fn(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clGetKernelSubGroupInfo(void* kernel, void* device, uint32_t param_name, size_t input_value_size, void* input_value, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, size_t, void*, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetKernelSubGroupInfo");
    if (!fn) return -1001;
    return fn(kernel, device, param_name, input_value_size, input_value, param_value_size, param_value, param_value_size_ret);
}

int32_t clWaitForEvents(uint32_t num_events, void* event_list) {
    typedef int32_t (*fn_t)(uint32_t, void*);
    fn_t fn = (fn_t)get_sym("clWaitForEvents");
    if (!fn) return -1001;
    return fn(num_events, event_list);
}

int32_t clGetEventInfo(void* event, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetEventInfo");
    if (!fn) return -1001;
    return fn(event, param_name, param_value_size, param_value, param_value_size_ret);
}

void* clCreateUserEvent(void* context, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateUserEvent");
    if (!fn) return NULL;
    return fn(context, errcode_ret);
}

int32_t clRetainEvent(void* event) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clRetainEvent");
    if (!fn) return -1001;
    return fn(event);
}

int32_t clReleaseEvent(void* event) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clReleaseEvent");
    if (!fn) return -1001;
    return fn(event);
}

int32_t clSetUserEventStatus(void* event, int32_t execution_status) {
    typedef int32_t (*fn_t)(void*, int32_t);
    fn_t fn = (fn_t)get_sym("clSetUserEventStatus");
    if (!fn) return -1001;
    return fn(event, execution_status);
}

int32_t clSetEventCallback(void* event, int32_t command_exec_callback_type, void* pfn_notify, void* user_data) {
    typedef int32_t (*fn_t)(void*, int32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clSetEventCallback");
    if (!fn) return -1001;
    return fn(event, command_exec_callback_type, pfn_notify, user_data);
}

int32_t clGetEventProfilingInfo(void* event, int32_t param_name, size_t param_value_size, void* param_value, void* param_value_size_ret) {
    typedef int32_t (*fn_t)(void*, int32_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clGetEventProfilingInfo");
    if (!fn) return -1001;
    return fn(event, param_name, param_value_size, param_value, param_value_size_ret);
}

int32_t clFlush(void* command_queue) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clFlush");
    if (!fn) return -1001;
    return fn(command_queue);
}

int32_t clFinish(void* command_queue) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clFinish");
    if (!fn) return -1001;
    return fn(command_queue);
}

int32_t clEnqueueReadBuffer(void* command_queue, void* buffer, uint32_t blocking_read, size_t offset, size_t cb, void* ptr, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, size_t, size_t, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueReadBuffer");
    if (!fn) return -1001;
    return fn(command_queue, buffer, blocking_read, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueReadBufferRect(void* command_queue, void* buffer, uint32_t blocking_read, void* buffer_origin, void* host_origin, void* region, size_t buffer_row_pitch, size_t buffer_slc_pitch, size_t host_row_pitch, size_t host_slc_pitch, void* ptr, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, void*, void*, void*, size_t, size_t, size_t, size_t, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueReadBufferRect");
    if (!fn) return -1001;
    return fn(command_queue, buffer, blocking_read, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slc_pitch, host_row_pitch, host_slc_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueWriteBuffer(void* command_queue, void* buffer, uint32_t blocking_write, size_t offset, size_t cb, void* ptr, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, size_t, size_t, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueWriteBuffer");
    if (!fn) return -1001;
    return fn(command_queue, buffer, blocking_write, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueWriteBufferRect(void* command_queue, void* buffer, uint32_t blocking_write, void* buffer_origin, void* host_origin, void* region, size_t buffer_row_pitch, size_t buffer_slc_pitch, size_t host_row_pitch, size_t host_slc_pitch, void* ptr, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, void*, void*, void*, size_t, size_t, size_t, size_t, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueWriteBufferRect");
    if (!fn) return -1001;
    return fn(command_queue, buffer, blocking_write, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slc_pitch, host_row_pitch, host_slc_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueFillBuffer(void* command_queue, void* buffer, void* pattern, size_t pattern_size, size_t offset, size_t size, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, size_t, size_t, size_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueFillBuffer");
    if (!fn) return -1001;
    return fn(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueCopyBuffer(void* command_queue, void* src_buffer, void* dst_buffer, size_t src_offset, size_t dst_offset, size_t cb, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, size_t, size_t, size_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueCopyBuffer");
    if (!fn) return -1001;
    return fn(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, cb, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueCopyBufferRect(void* command_queue, void* src_buffer, void* dst_buffer, void* src_origin, void* dst_origin, void* region, size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, void*, void*, void*, size_t, size_t, size_t, size_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueCopyBufferRect");
    if (!fn) return -1001;
    return fn(command_queue, src_buffer, dst_buffer, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueReadImage(void* command_queue, void* image, uint32_t blocking_read, void* origin, void* region, size_t row_pitch, size_t slice_pitch, void* ptr, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, void*, void*, size_t, size_t, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueReadImage");
    if (!fn) return -1001;
    return fn(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueWriteImage(void* command_queue, void* image, uint32_t blocking_write, void* origin, void* region, size_t input_row_pitch, size_t input_slice_pitch, void* ptr, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, void*, void*, size_t, size_t, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueWriteImage");
    if (!fn) return -1001;
    return fn(command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueFillImage(void* command_queue, void* image, void* fill_color, void* origin, void* region, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueFillImage");
    if (!fn) return -1001;
    return fn(command_queue, image, fill_color, origin, region, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueCopyImage(void* command_queue, void* src_image, void* dst_image, void* src_origin, void* dst_origin, void* region, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, void*, void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueCopyImage");
    if (!fn) return -1001;
    return fn(command_queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueCopyImageToBuffer(void* command_queue, void* src_image, void* dst_buffer, void* src_origin, void* region, size_t dst_offset, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, void*, void*, size_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueCopyImageToBuffer");
    if (!fn) return -1001;
    return fn(command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueCopyBufferToImage(void* command_queue, void* src_buffer, void* dst_image, size_t src_offset, void* dst_origin, void* region, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, size_t, void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueCopyBufferToImage");
    if (!fn) return -1001;
    return fn(command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}

void* clEnqueueMapBuffer(void* command_queue, void* buffer, uint32_t blocking_map, uint32_t map_flags, size_t offset, size_t size, uint32_t num_events_in_wait_list, void* event_wait_list, void* event, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*, uint32_t, uint32_t, size_t, size_t, uint32_t, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueMapBuffer");
    if (!fn) return NULL;
    return fn(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

void* clEnqueueMapImage(void* command_queue, void* image, uint32_t blocking_map, uint32_t map_flags, void* origin, void* region, void* image_row_pitch, void* image_slice_pitch, uint32_t num_events_in_wait_list, void* event_wait_list, void* event, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*, uint32_t, uint32_t, void*, void*, void*, void*, uint32_t, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueMapImage");
    if (!fn) return NULL;
    return fn(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

int32_t clEnqueueUnmapMemObject(void* command_queue, void* memobj, void* mapped_ptr, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueUnmapMemObject");
    if (!fn) return -1001;
    return fn(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueMigrateMemObjects(void* command_queue, uint32_t num_mem_objects, void* mem_objects, void* flags, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueMigrateMemObjects");
    if (!fn) return -1001;
    return fn(command_queue, num_mem_objects, mem_objects, flags, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueNDRangeKernel(void* command_queue, void* kernel, uint32_t work_dim, void* global_work_offset, void* global_work_dims, void* local_work_dims, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, void*, void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueNDRangeKernel");
    if (!fn) return -1001;
    return fn(command_queue, kernel, work_dim, global_work_offset, global_work_dims, local_work_dims, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueNativeKernel(void* command_queue, void* user_func, void* args, size_t cb_args, uint32_t num_mem_objects, void* mem_list, void** args_mem_loc, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, size_t, uint32_t, void*, void**, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueNativeKernel");
    if (!fn) return -1001;
    return fn(command_queue, user_func, args, cb_args, num_mem_objects, mem_list, args_mem_loc, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueMarkerWithWaitList(void* command_queue, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueMarkerWithWaitList");
    if (!fn) return -1001;
    return fn(command_queue, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueBarrierWithWaitList(void* command_queue, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueBarrierWithWaitList");
    if (!fn) return -1001;
    return fn(command_queue, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueSVMFree(void* command_queue, uint32_t num_svm_pointers, void** svm_pointers, void* pfn_free_func, void* user_data, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, uint32_t, void**, void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueSVMFree");
    if (!fn) return -1001;
    return fn(command_queue, num_svm_pointers, svm_pointers, pfn_free_func, user_data, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueSVMMemcpy(void* command_queue, uint32_t blocking_copy, void* dst_ptr, void* src_ptr, size_t size, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*, void*, size_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueSVMMemcpy");
    if (!fn) return -1001;
    return fn(command_queue, blocking_copy, dst_ptr, src_ptr, size, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueSVMMemFill(void* command_queue, void* svm_ptr, void* pattern, size_t pattern_size, size_t size, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, void*, size_t, size_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueSVMMemFill");
    if (!fn) return -1001;
    return fn(command_queue, svm_ptr, pattern, pattern_size, size, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueSVMMap(void* command_queue, uint32_t blocking_map, uint32_t flags, void* svm_ptr, size_t size, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, uint32_t, uint32_t, void*, size_t, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueSVMMap");
    if (!fn) return -1001;
    return fn(command_queue, blocking_map, flags, svm_ptr, size, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueSVMUnmap(void* command_queue, void* svm_ptr, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueSVMUnmap");
    if (!fn) return -1001;
    return fn(command_queue, svm_ptr, num_events_in_wait_list, event_wait_list, event);
}

int32_t clEnqueueSVMMigrateMem(void* command_queue, uint32_t num_svm_pointers, void** svm_pointers, void* sizes, void* flags, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, uint32_t, void**, void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueSVMMigrateMem");
    if (!fn) return -1001;
    return fn(command_queue, num_svm_pointers, svm_pointers, sizes, flags, num_events_in_wait_list, event_wait_list, event);
}

void* clGetExtensionFunctionAddressForPlatform(void* platform, char* func_name) {
    typedef void* (*fn_t)(void*, char*);
    fn_t fn = (fn_t)get_sym("clGetExtensionFunctionAddressForPlatform");
    if (!fn) return NULL;
    return fn(platform, func_name);
}

void* clCreateImage2D(void* context, uint64_t flags, void* image_format, size_t image_width, size_t image_depth, size_t image_row_pitch, void* host_ptr, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint64_t, void*, size_t, size_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateImage2D");
    if (!fn) return NULL;
    return fn(context, flags, image_format, image_width, image_depth, image_row_pitch, host_ptr, errcode_ret);
}

void* clCreateImage3D(void* context, uint64_t flags, void* image_format, size_t image_width, size_t image_height, size_t image_depth, size_t image_row_pitch, size_t image_slice_pitch, void* host_ptr, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint64_t, void*, size_t, size_t, size_t, size_t, size_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateImage3D");
    if (!fn) return NULL;
    return fn(context, flags, image_format, image_width, image_height, image_depth, image_row_pitch, image_slice_pitch, host_ptr, errcode_ret);
}

int32_t clEnqueueMarker(void* command_queue, void* event) {
    typedef int32_t (*fn_t)(void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueMarker");
    if (!fn) return -1001;
    return fn(command_queue, event);
}

int32_t clEnqueueWaitForEvents(void* command_queue, uint32_t num_events, void* event_list) {
    typedef int32_t (*fn_t)(void*, uint32_t, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueWaitForEvents");
    if (!fn) return -1001;
    return fn(command_queue, num_events, event_list);
}

int32_t clEnqueueBarrier(void* command_queue) {
    typedef int32_t (*fn_t)(void*);
    fn_t fn = (fn_t)get_sym("clEnqueueBarrier");
    if (!fn) return -1001;
    return fn(command_queue);
}

int32_t clUnloadCompiler(void) {
    typedef int32_t (*fn_t)(void);
    fn_t fn = (fn_t)get_sym("clUnloadCompiler");
    if (!fn) return -1001;
    return fn();
}

void clGetExtensionFunctionAddress(char* func_name) {
    typedef void (*fn_t)(char*);
    fn_t fn = (fn_t)get_sym("clGetExtensionFunctionAddress");
    if (fn) fn(func_name);
}

void* clCreateCommandQueue(void* context, void* device, void* properties, void* errcode_ret) {
    typedef void* (*fn_t)(void*, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateCommandQueue");
    if (!fn) return NULL;
    return fn(context, device, properties, errcode_ret);
}

void* clCreateSampler(void* context, uint32_t normalize_coords, void* addressing_mode, void* filter_mode, void* errcode_ret) {
    typedef void* (*fn_t)(void*, uint32_t, void*, void*, void*);
    fn_t fn = (fn_t)get_sym("clCreateSampler");
    if (!fn) return NULL;
    return fn(context, normalize_coords, addressing_mode, filter_mode, errcode_ret);
}

int32_t clEnqueueTask(void* command_queue, void* kernel, uint32_t num_events_in_wait_list, void* event_wait_list, void* event) {
    typedef int32_t (*fn_t)(void*, void*, uint32_t, void*, void*);
    fn_t fn = (fn_t)get_sym("clEnqueueTask");
    if (!fn) return -1001;
    return fn(command_queue, kernel, num_events_in_wait_list, event_wait_list, event);
}

#endif /* __linux__ */
