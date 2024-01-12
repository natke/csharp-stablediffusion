
# Inference Stable Diffusion with C# and ONNX Runtime

This repo contains a console app to run the popular Stable Diffusion deep learning model in C#.  Stable Diffusion models take a text prompt and create an image that represents the text.

You can also provide a negative prompt, which excludes the content described.

This sample uses the OnnxStack library, which builds upon the ONNX Runtime C# API.

## Models

To run this sample, you need to download the ONNX format models for the Stable Diffusion pipeline that you want to run.

The `appsettings.json` file, which is loaded at runtime, has example configs for Stable Diffusion and Stable Diffusion XL.

Everything in the `models` folder is copied to the output folder when the project is built. You can add your models there, and reference them in the appsettings.json file.

## Build and run 

Install .NET 7 if you don't alreadu have it installed.

Either load the StableDiffusion.sln file into Visual Studio and build and run.

Or run

```bash
dotnet build
dotnet run
```

## Configuration



