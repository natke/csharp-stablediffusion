using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using SixLabors.ImageSharp;
using System;

internal class StableDiffusionRunner : IHostedService
{
    private readonly string _outputDirectory;
    private readonly IStableDiffusionService _stableDiffusionService;
    private readonly StableDiffusionConfig _stableDiffusionConfig;

    public StableDiffusionRunner(IConfiguration configuration, IStableDiffusionService stableDiffusionService)
    {
        _stableDiffusionService = stableDiffusionService;
        _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Images");
        _stableDiffusionConfig = new();
        configuration.GetSection("StableDiffusionConfig").Bind(_stableDiffusionConfig);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(_outputDirectory);
        var modelSets = _stableDiffusionConfig.ModelSets;

        System.Console.WriteLine("Choose from the following examples:");
        for (int i=0; i < modelSets.Count; i++)
        {
            System.Console.WriteLine($"{i + 1}: {modelSets[i].Name}");
        }
        var index = 1;
        var selection = System.Console.ReadLine();
        if (!int.TryParse(selection, out index) || index > modelSets.Count)
        {
            System.Console.WriteLine($"{selection} is an invalid option number");
        }
        var modelSet = _stableDiffusionConfig.ModelSets[index-1];

        System.Console.WriteLine($"Loading models for {modelSet.Name}...");
        await _stableDiffusionService.LoadModelAsync(modelSet);

        while (true)
        {
            System.Console.WriteLine("Please type a prompt and press ENTER");
            var prompt = System.Console.ReadLine();

            System.Console.WriteLine("Please type a negative prompt and press ENTER (optional)");
            var negativePrompt = System.Console.ReadLine();

            ModelOptions modelOptions = new ModelOptions(modelSet);

            PromptOptions promptOptions = new PromptOptions
            {
                Prompt = prompt,
                NegativePrompt = negativePrompt,
                DiffuserType = DiffuserType.TextToImage
            };

            var schedulerOptions = new SchedulerOptions
            {
                Seed = Random.Shared.Next(),
                SchedulerType = SchedulerType.EulerAncestral
            };

            // Generate Image Example
            var outputFilename = Path.Combine(_outputDirectory, $"{prompt}-without-{negativePrompt}-{schedulerOptions.Seed}.png");
            Image result = await _stableDiffusionService.GenerateAsImageAsync(modelOptions, promptOptions, schedulerOptions);
            if (result is not null)
            {
                // Save image to disk
                Console.WriteLine("Saving image: " + outputFilename + " to disk");
                await result.SaveAsPngAsync(outputFilename);
            }
        }

    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }
}