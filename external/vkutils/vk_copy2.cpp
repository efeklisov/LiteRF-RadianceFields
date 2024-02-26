#include "vk_copy2.h"
#include "vk_alloc.h"
#include "vk_utils.h"
#include "vk_buffers.h"


namespace vk_utils
{
  PingPongCopyHelper2::PingPongCopyHelper2(VkDevice a_device, VkPhysicalDevice a_physDevice,
                                           std::shared_ptr<IMemoryAlloc> a_pAlloc,
                                           uint32_t a_queueIDX, size_t a_stagingBuffSize)

  {
    physDev  = a_physDevice;
    dev      = a_device;
    pAlloc   = a_pAlloc;
    vkGetDeviceQueue(a_device, a_queueIDX, 0, &queue);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = a_queueIDX;
    VK_CHECK_RESULT(vkCreateCommandPool(a_device, &poolInfo, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cmdBufAllocInfo = {};
    cmdBufAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool        = cmdPool;
    cmdBufAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &cmdBufAllocInfo, &cmdBuff));

    stagingSize     = a_stagingBuffSize;
    stagingSizeHalf = a_stagingBuffSize/2;

    VkBufferUsageFlags    usage    = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    staging[0] =  vk_utils::createBuffer(dev, stagingSizeHalf, usage);
    staging[1] =  vk_utils::createBuffer(dev, stagingSizeHalf, usage);

    MemAllocInfo allocInfo{};
    allocInfo.memUsage = memProps;

    std::vector<VkBuffer> tmpVec = {staging[0], staging[1]};
    allocId = a_pAlloc->Allocate(allocInfo, tmpVec);

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));
  }

  PingPongCopyHelper2::~PingPongCopyHelper2()
  {
    pAlloc->Free(allocId);
  }

}