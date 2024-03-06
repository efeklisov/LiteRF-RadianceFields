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

const size_t OUTER_LOOP = 15;
const size_t IMAGE_LOOP = 7;

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

  const size_t gridSize = 128;
  pImpl->InitGrid(gridSize);
  pImpl->optimizerInit();
  pImpl_d->InitGrid(gridSize);

  //pImpl->SetBoundingBox(float3(-0.5, -0.5, -0.5), float3(0.5, 0.5, 0.5));
  pImpl->SetBoundingBox(float3(0, 0, 0), float3(1, 1, 1));
  // pImpl->LoadModel("../sigmat.json", "../sh_coeffs.json");

  json cam_data;
  {
    std::ifstream f("../data/radiance_fields/lego/train_camera_params.json");
    cam_data = json::parse(f);
  }

  stbi_set_flip_vertically_on_load(true);

  for (int j = 0; j < OUTER_LOOP; j++) {
    // int i = 0;
    // for(const auto & cam : cam_data) 
    // {
      const int WIN_WIDTH = 256;
      const int WIN_HEIGHT = 256;
      
      // const int WIN_WIDTH = cam["intrinsic"]["width"];
      // const int WIN_HEIGHT = cam["intrinsic"]["height"];

      std::vector<uint> pixelData(WIN_WIDTH * WIN_HEIGHT);  

      // float4x4 viewMat;
      // viewMat[0][0] = std::stof(cam["extrinsic"]["rotation"][0][0].get<std::string>());
      // viewMat[0][1] = std::stof(cam["extrinsic"]["rotation"][0][1].get<std::string>());
      // viewMat[0][2] = std::stof(cam["extrinsic"]["rotation"][0][2].get<std::string>());

      // viewMat[1][0] = std::stof(cam["extrinsic"]["rotation"][1][0].get<std::string>());
      // viewMat[1][1] = std::stof(cam["extrinsic"]["rotation"][1][1].get<std::string>());
      // viewMat[1][2] = std::stof(cam["extrinsic"]["rotation"][1][2].get<std::string>());

      // viewMat[2][0] = std::stof(cam["extrinsic"]["rotation"][2][0].get<std::string>());
      // viewMat[2][1] = std::stof(cam["extrinsic"]["rotation"][2][1].get<std::string>());
      // viewMat[2][2] = std::stof(cam["extrinsic"]["rotation"][2][2].get<std::string>());

      // viewMat[0][3] = std::stof(cam["extrinsic"]["translation"][0][0].get<std::string>());
      // viewMat[1][3] = std::stof(cam["extrinsic"]["translation"][1][0].get<std::string>());
      // viewMat[2][3] = std::stof(cam["extrinsic"]["translation"][2][0].get<std::string>());

      // viewMat[3][3] = 1.0;

      // pImpl->SetWorldViewMatrix(transpose(viewMat));

      // focal = 0.5 * width / tan(0.5 * fov)
      // width / (2 * focal) = tan(0.5 * fov)
      // fov = 2 * arctan(width / (2 * focal))
      // const float FOV = 2 * atan(WIN_WIDTH / (2 * (float)cam["intrinsic"]["focal"])) * 180.0 / M_PI;


      pImpl->SetWorldViewMProjatrix(perspectiveMatrix(45, 1, 0.1, 100));

      pImpl->UpdateMembersPlainData();                                            // copy all POD members from CPU to GPU in GPU implementation
    
      float timings[4] = {0,0,0,0};
      pImpl->GetExecutionTime("RayMarch", timings);

      // LiteImage::SaveBMP("buffer.bmp", input, WIDTH, HEIGHT);
      
      pImpl_d->zeroGrad();
      for (int k = 0; k < IMAGE_LOOP; k++) {
        std::stringstream inputImgStrStream;
        inputImgStrStream << "../data/r_" << k << ".png";
        std::string inputImgStr = inputImgStrStream.str();

        int WIDTH = WIN_WIDTH;
        int HEIGHT = WIN_HEIGHT;
        int CHANNELS = 4;

        float* input_raw = stbi_loadf(inputImgStr.c_str(), &WIDTH, &HEIGHT, &CHANNELS, CHANNELS);
        float4* input = (float4*)input_raw;

        // float4x4 viewMat =  translate4x4(float3(0.5, 0.5, 0.5)) * rotate4x4Y(float(360.0 / 7 * k)*DEG_TO_RAD) * lookAt(float3(0.0, 0.0, 1.3), float3(0.0, 0.0, 0.0), float3(0.0, 1.0, 0.0));
        float4x4 viewMat =  lookAt(float3(0.0, 0.0, 1.3), float3(0.0, 0.0, 0.0), float3(0.0, 1.0, 0.0)) * rotate4x4Y(-float(360.0 / 7 * k)*DEG_TO_RAD) * translate4x4(float3(-0.5, -0.5, -0.5));
        pImpl->SetWorldViewMatrix(viewMat);

        std::stringstream strOut;
        strOut << std::fixed << std::setprecision(2) << "out_cpu_" << k << ".bmp";
        std::string fileName = strOut.str();

        float loss, loss_d;

        // std::vector<float4> L(WIN_WIDTH * WIN_HEIGHT);
        // pImpl->kernel2D_RayMarch(pixelData.data(), WIN_WIDTH, WIN_HEIGHT, L.data());

        // LiteImage::SaveBMP(fileName.c_str(), pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

        // std::fill(pixelData.begin(), pixelData.end(), 0);
        L1Loss(&loss, input, pixelData.data(), WIN_WIDTH, WIN_HEIGHT, pImpl.get(), pImpl_d.get(), fileName.c_str());
        
        std::cout << loss << ' ' << loss_d << std::endl;
        free(input);
      }
      pImpl->optimizerStep(pImpl_d.get(), j);

    //   i++;

    //   if (i == 10) break;
    // }
  }
  
  pImpl = nullptr;
  return 0;
}