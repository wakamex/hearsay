// The Vulkan loader opened at run time instead of linked, for -Dgpu=vulkan builds: a program with
// hearsay then starts on a computer without Vulkan and runs on the CPU there. ggml reaches Vulkan
// through a dispatcher filled from vkGetInstanceProcAddr, and calls the four functions below
// directly; each is the loader's own, looked up when the program starts. Without a loader,
// GGML_DISABLE_VULKAN is set before ggml registers its backends, so it never tries the GPU. An app
// can set GGML_DISABLE_VULKAN itself to keep to the CPU.

#include <stdlib.h>
#include <vulkan/vulkan_core.h>

#ifdef _WIN32
#include <windows.h>
static void * open_loader() { return (void *) LoadLibraryA("vulkan-1.dll"); }
static void * find(void * lib, const char * name) { return (void *) GetProcAddress((HMODULE) lib, name); }
#else
#include <dlfcn.h>
static void * open_loader() {
    void * lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    return lib ? lib : dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
}
static void * find(void * lib, const char * name) { return dlsym(lib, name); }
#endif

static PFN_vkGetInstanceProcAddr        get_instance_proc_addr;
static PFN_vkGetDeviceProcAddr          get_device_proc_addr;
static PFN_vkGetPhysicalDeviceFeatures2 get_physical_device_features2;
static PFN_vkCmdCopyBuffer              cmd_copy_buffer;

static bool loaded = [] {
    void * lib = getenv("GGML_DISABLE_VULKAN") ? nullptr : open_loader();
    if (lib) {
        get_instance_proc_addr        = (PFN_vkGetInstanceProcAddr) find(lib, "vkGetInstanceProcAddr");
        get_device_proc_addr          = (PFN_vkGetDeviceProcAddr) find(lib, "vkGetDeviceProcAddr");
        get_physical_device_features2 = (PFN_vkGetPhysicalDeviceFeatures2) find(lib, "vkGetPhysicalDeviceFeatures2");
        cmd_copy_buffer               = (PFN_vkCmdCopyBuffer) find(lib, "vkCmdCopyBuffer");
    }
    bool ok = get_instance_proc_addr && get_device_proc_addr && get_physical_device_features2 && cmd_copy_buffer;
    if (!ok) {
#ifdef _WIN32
        _putenv_s("GGML_DISABLE_VULKAN", "1");
#else
        setenv("GGML_DISABLE_VULKAN", "1", 1);
#endif
    }
    return ok;
}();

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char * pName) {
    return loaded ? get_instance_proc_addr(instance, pName) : nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char * pName) {
    return loaded ? get_device_proc_addr(device, pName) : nullptr;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 * pFeatures) {
    get_physical_device_features2(physicalDevice, pFeatures);
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy * pRegions) {
    cmd_copy_buffer(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
}

}
