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
std::shared_ptr<RayMarcherExample> CreateRayMarcherExample_Generated(vk_utils::VulkanContext a_ctx,
  size_t a_maxThreadsGenerated);
#endif

const size_t UPSAMPLE_LOOP = 4;
const size_t OUTER_LOOP = 15;
const size_t IMAGE_LOOP = 7;

int main(int argc, const char **argv)
{
#ifndef NDEBUG
  bool enableValidationLayers = true;
#else
  bool enableValidationLayers = false;
#endif

  std::shared_ptr<RayMarcherExample> pImpl = nullptr;
#ifdef USE_VULKAN
  bool onGPU = true; // TODO: you can read it from command line
  if (onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, 0);
    pImpl = CreateRayMarcherExample_Generated(ctx, WIN_WIDTH * WIN_HEIGHT);
  }
  else
#else
  bool onGPU = false;
#endif
    pImpl = std::make_shared<RayMarcherExample>();

  pImpl->CommitDeviceData();

  const size_t gridSize = 16;
  pImpl->InitGrid(gridSize);
  pImpl->optimizerInit();
  pImpl->InitGrad();

  // std::ifstream fout("model.dat", std::ios::in | std::ios::binary);
  // fout.read((char*)&pImpl->grid[0], pImpl->grid.size() * sizeof(Cell));
  // fout.close();

  pImpl->SetBoundingBox(float3(0, 0, 0), float3(1, 1, 1));
  // pImpl->LoadModel("../sigmat.json", "../sh_coeffs.json");

  json cam_data;
  {
    std::ifstream f("../data/radiance_fields/lego/train_camera_params.json");
    cam_data = json::parse(f);
  }

  const int WIN_WIDTH = 256;
  const int WIN_HEIGHT = 256;

  std::vector<uint> pixelData(WIN_WIDTH * WIN_HEIGHT);

  stbi_set_flip_vertically_on_load(true);

  for (int i = 0; i < UPSAMPLE_LOOP; i++)
  {
    for (int j = 0; j < OUTER_LOOP; j++)
    {
      pImpl->SetWorldViewMProjatrix(perspectiveMatrix(45, 1, 0.1, 100));

      pImpl->UpdateMembersPlainData(); // copy all POD members from CPU to GPU in GPU implementation

      float timings[4] = {0, 0, 0, 0};
      pImpl->GetExecutionTime("RayMarch", timings);

      pImpl->zeroGrad();
      for (int k = 0; k < IMAGE_LOOP; k++)
      {
        std::stringstream inputImgStrStream;
        inputImgStrStream << "../data/r_" << k << ".png";
        std::string inputImgStr = inputImgStrStream.str();

        int WIDTH = WIN_WIDTH;
        int HEIGHT = WIN_HEIGHT;
        int CHANNELS = 4;

        float *input_raw = stbi_loadf(inputImgStr.c_str(), &WIDTH, &HEIGHT, &CHANNELS, CHANNELS);
        float4 *input = (float4 *)input_raw;

        float4x4 viewMat = 
          lookAt(float3(0.0, 0.0, 1.3), float3(0.0, 0.0, 0.0), float3(0.0, 1.0, 0.0)) * 
          rotate4x4Y(-float(360.0 / IMAGE_LOOP * k) * DEG_TO_RAD) * 
          translate4x4(float3(-0.5, -0.5, -0.5));

        pImpl->SetWorldViewMatrix(viewMat);

        std::stringstream strOut;
        strOut << std::fixed << std::setprecision(2) << "out_cpu_" << 
          pImpl->gridSize << '_' << k << ".bmp";
        std::string fileName = strOut.str();


        float loss, loss_d;

        L1Loss(&loss, input, pixelData.data(), WIN_WIDTH, WIN_HEIGHT, pImpl.get(), 
          fileName.c_str());

        std::cout << loss << ' ' << loss_d << std::endl;
        free(input);
      }

      pImpl->kernel2D_TVGrad();
      pImpl->optimizerStep(j);
    }

    if (i < UPSAMPLE_LOOP - 1)
    {
      std::cout << "Upsample start" << std::endl;
      pImpl->UpsampleGrid();
      pImpl->InitGrad();
      pImpl->optimizerInit();
      pImpl->UpsampleOctree();
      std::cout << "Upsample end" << std::endl;
    }
  }

  std::cout << pImpl->boxes.size() << ' ' << 
    ((pImpl->gridSize - 1) * (pImpl->gridSize - 1) * (pImpl->gridSize - 1)) << std::endl;

  // std::ofstream fout("model.dat", std::ios::out | std::ios::binary);
  // fout.write((char*)&pImpl->grid[0], pImpl->grid.size() * sizeof(Cell));
  // fout.close();
  std::cout << "Cleanup start" << std::endl;
  pImpl = nullptr;
  std::cout << "Cleanup end" << std::endl;
  return 0;
}