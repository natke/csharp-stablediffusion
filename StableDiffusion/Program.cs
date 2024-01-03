using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] args)
        {


        //test how long this takes to execute
        var watch = System.Diagnostics.Stopwatch.StartNew();

            //Default args
            var prompt = "a fireplace in an old cabin in the woods";
            var negativePrompt = "";
            Console.WriteLine(prompt);
            Console.WriteLine(negativePrompt);

            var config = new StableDiffusionConfig
            {
                // Set stable diffusion XL variant
                isStableDiffusionXL = true,

                // Number of denoising steps
                NumInferenceSteps = 5,
                // Scale for classifier-free guidance
                GuidanceScale = 7.5,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cpu,
                // Set GPU Device ID.
                DeviceId = 0,
                // Update paths to your models
                TokenizerOnnxPath = @"models\tokenizer\cliptokenizer.onnx",
                TextEncoderOnnxPath = @"models\text_encoder\model.onnx",
                TextEncoder2OnnxPath = @"models\text_encoder_2\model.onnx",
                UnetOnnxPath = @"models\unet\model.onnx",
                VaeDecoderOnnxPath = @"models\vae_decoder\model.onnx"
            };

            // Select which pipeline to run
            var pipeline = new StableDiffusionXLPipeline(config);

            var image = pipeline.Run(prompt, negativePrompt, config);

            if (image == null)
            {
                Console.WriteLine("Unable to create image, please try again.");
            }

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Time taken: " + elapsedMs + "ms");

        }

    }
}