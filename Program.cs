using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OnnxStack.Core;

namespace StableDiffusionConsoleApp
{
    internal class Program
    {
        static async Task Main(string[] _)
        {
            Console.CancelKeyPress += delegate(object? sender, ConsoleCancelEventArgs e) {
                e.Cancel = true;
                System.Console.WriteLine("Exiting...");
                System.Environment.Exit(0);
            };
        
            var builder = Host.CreateApplicationBuilder();
            builder.Logging.ClearProviders();
            builder.Services.AddLogging((loggingBuilder) => loggingBuilder.SetMinimumLevel(LogLevel.Error));

            // Add OnnxStack Stable Diffusion
            builder.Services.AddOnnxStackStableDiffusion();

            // Add AppService
            builder.Services.AddHostedService<StableDiffusionRunner>();
            
            // Start
            var app = builder.Build();

            // Ask the service provider for the configuration abstraction.
            IConfiguration config = app.Services.GetRequiredService<IConfiguration>();

            await app.RunAsync();
        }

    }
}