#include <iostream>
#include <fstream>
#include <vector>
#include <memory>  // for shared pointers
#include <iomanip> // for std::fixed/std::setprecision
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "example_tracer/example_tracer.h"
#include "Image2d.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<RayMarcherExample> CreateRayMarcherExample_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::shared_ptr<RayMarcherExample> pImpl = nullptr;
  std::shared_ptr<RayMarcherExample> pImpl_d = nullptr;
  #ifdef USE_VULKAN
  bool onGPU = true; // TODO: you can read it from command line
  if(onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, 0);
    pImpl    = CreateRayMarcherExample_Generated(ctx, WIN_WIDTH*WIN_HEIGHT);
  }
  else
  #else
  bool onGPU = false;
  #endif
    pImpl = std::make_shared<RayMarcherExample>();
    pImpl_d = std::make_shared<RayMarcherExample>();

  pImpl->CommitDeviceData();

  const size_t gridSize = 16;
  pImpl->InitGrid(gridSize);
  pImpl_d->InitGrid(gridSize);

  pImpl->SetBoundingBox(float3(-0.5, -0.5, -0.5), float3(0.5, 0.5, 0.5));
  // pImpl->LoadModel("../sigmat.json", "../sh_coeffs.json");

  json cam_data;
  {
    std::ifstream f("../data/radiance_fields/lego/train_camera_params.json");
    cam_data = json::parse(f);
  }

  int i = 0;
  for(const auto & cam : cam_data) 
  {
    const int WIN_WIDTH = cam["intrinsic"]["width"];
    const int WIN_HEIGHT = cam["intrinsic"]["height"];

    std::vector<uint> pixelData(WIN_WIDTH * WIN_HEIGHT);  

    float4x4 viewMat;
    viewMat[0][0] = std::stof(cam["extrinsic"]["rotation"][0][0].get<std::string>());
    viewMat[0][1] = std::stof(cam["extrinsic"]["rotation"][0][1].get<std::string>());
    viewMat[0][2] = std::stof(cam["extrinsic"]["rotation"][0][2].get<std::string>());

    viewMat[1][0] = std::stof(cam["extrinsic"]["rotation"][1][0].get<std::string>());
    viewMat[1][1] = std::stof(cam["extrinsic"]["rotation"][1][1].get<std::string>());
    viewMat[1][2] = std::stof(cam["extrinsic"]["rotation"][1][2].get<std::string>());

    viewMat[2][0] = std::stof(cam["extrinsic"]["rotation"][2][0].get<std::string>());
    viewMat[2][1] = std::stof(cam["extrinsic"]["rotation"][2][1].get<std::string>());
    viewMat[2][2] = std::stof(cam["extrinsic"]["rotation"][2][2].get<std::string>());

    viewMat[0][3] = std::stof(cam["extrinsic"]["translation"][0][0].get<std::string>());
    viewMat[1][3] = std::stof(cam["extrinsic"]["translation"][1][0].get<std::string>());
    viewMat[2][3] = std::stof(cam["extrinsic"]["translation"][2][0].get<std::string>());

    viewMat[3][3] = 1.0;

    pImpl->SetWorldViewMatrix(transpose(viewMat));
    pImpl->SetWorldViewMProjatrix(perspectiveMatrix(120, 1, 1, 8));

    pImpl->UpdateMembersPlainData();                                            // copy all POD members from CPU to GPU in GPU implementation
    // pImpl->RayMarch(pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
    float timings[4] = {0,0,0,0};
    pImpl->GetExecutionTime("RayMarch", timings);

    std::stringstream inputImgStrStream;
    inputImgStrStream << "../data/radiance_fields/lego/train/r_" << i << ".png";
    std::string inputImgStr = inputImgStrStream.str();

    int WIDTH = WIN_WIDTH;
    int HEIGHT = WIN_HEIGHT;
    int CHANNELS = 4;

    stbi_uc* input_raw = stbi_load(inputImgStr.c_str(), &WIDTH, &HEIGHT, &CHANNELS, CHANNELS);
    uint* input = (uint*)input_raw;                                                                         

    for (int j = 0; j < 20; j++) {
      std::stringstream strOut;
      strOut << std::fixed << std::setprecision(2) << "out_cpu_" << i << "_step_" << j << ".bmp";
      std::string fileName = strOut.str();

      float loss, loss_d;

      pImpl_d->zeroGrad();
      L1Loss(&loss, input, pixelData.data(), WIN_WIDTH, WIN_HEIGHT, pImpl.get(), pImpl_d.get(), fileName.c_str());
      pImpl->optimizerStep(pImpl_d.get(), 1.f);

      std::cout << loss << ' ' << loss_d << std::endl;
    }

    // for (int y = 0; y < WIN_HEIGHT; y++)
    //   for(int x = 0; x < WIN_WIDTH; x++)
    //     std::cout << gen_d[y * WIN_WIDTH + x] << ' ';

    // for (int z = 0; z < gridSize; z++)
    //   for (int y = 0; y < gridSize; y++)
    //     for (int x = 0; x < gridSize; x++)
    //       if (pImpl->grid[z* gridSize * gridSize + y * gridSize + x].density != 0.0f)
    //         std::cout << pImpl->grid[z* gridSize * gridSize + y * gridSize + x].density << ' ';

    i++;
  }
  
  pImpl = nullptr;
  return 0;
}