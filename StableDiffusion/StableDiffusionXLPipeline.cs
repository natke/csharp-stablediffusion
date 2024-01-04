using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using MathNet.Numerics.Distributions;

namespace StableDiffusion
{
    internal class StableDiffusionXLPipeline
    {
        OrtValue placeholder;

        InferenceSession TokenizerSession;
        
        InferenceSession Tokenizer2Session;
        
        InferenceSession TextEmbeddingSession;
        
        InferenceSession TextEmbedding2Session;
        
        InferenceSession UNetSession;

        InferenceSession VaeDecoderSession;


        public StableDiffusionXLPipeline(StableDiffusionConfig config)
        {
            // Create session for the tokenizer
            var tokenizerSessionOptions = new SessionOptions();
            tokenizerSessionOptions.RegisterOrtExtensions();

            // Create an InferenceSession from the onnx clip tokenizer.
            TokenizerSession = new InferenceSession(config.TokenizerOnnxPath, tokenizerSessionOptions);

            // Create an InferenceSession from the onnx clip tokenizer.
            Tokenizer2Session = new InferenceSession(config.TokenizerOnnxPath, tokenizerSessionOptions);

            // Create a session for the first text embedding model
            var sessionOptions = config.GetSessionOptionsForEp();
            TextEmbeddingSession = new InferenceSession(config.TextEncoderOnnxPath, sessionOptions);

            // Create a session for the second text embedding model
            TextEmbedding2Session = new InferenceSession(config.TextEncoder2OnnxPath, sessionOptions);

            // Create a session for the UNet model
            UNetSession = new InferenceSession(config.UnetOnnxPath, sessionOptions);

            // Create a session for the Vae decoder model
            VaeDecoderSession = new InferenceSession(config.VaeDecoderOnnxPath, sessionOptions);
        }

        public IDictionary<String, OrtValue> EncodePrompt(string prompt, InferenceSession tokenizerSession, InferenceSession textEmbeddingSession)
        {
            //
            // Encode string prompt with first tokenizer, required as input to the first text embedding model
            //
            long[] promptShape = { 1 };
            using var promptTensor = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, promptShape);
            promptTensor.StringTensorSetElementAt(prompt.AsSpan(), 0);

            // Set the inputs
            var tokenizerInputs = new Dictionary<string, OrtValue>();
            tokenizerInputs["string_input"] = promptTensor;

            // Run the tokenizer session
            using var runOptions = new RunOptions();
            var outputs = tokenizerSession.Run(runOptions, tokenizerInputs, tokenizerSession.OutputNames);

            // Set the outputs
            var tokenizerOutputs = new Dictionary<string, OrtValue>();
            for (int i = 0; i < outputs.Count; i++)
            {
                tokenizerOutputs[tokenizerSession.OutputNames[i]] = outputs[i];
            }
            var inputIds = tokenizerOutputs["input_ids"];
            var inputIdsShape = inputIds.GetTensorTypeAndShape().Shape;

            // Check to see if the text embedding model requires int32 token ids.
            // If so, cast from the int64 ones.
            if (textEmbeddingSession.InputMetadata["input_ids"].ElementDataType == TensorElementType.Int32)
            {
                // Create a span over the data in the outputs
                var data = inputIds.GetTensorDataAsSpan<long>();

                // Cast longs to int
                var inputIdsInt = new int[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    inputIdsInt[i] = i;
                }

                // Create OrtValue from array
                inputIds = OrtValue.CreateTensorValueFromMemory<int>(inputIdsInt, inputIdsShape);
            }

            //
            // Create text embeddings from the token ids
            //

            // Set the inputs
            var textEmbeddingInputs = new Dictionary<string, OrtValue>();
            textEmbeddingInputs["input_ids"] = inputIds;

            // Run the text embedding session
            outputs = textEmbeddingSession.Run(runOptions, textEmbeddingInputs, textEmbeddingSession.OutputNames);

            // Set the outputs
            var textEmbeddingOutputs = new Dictionary<string, OrtValue>();
            for (int i = 0; i < outputs.Count; i++)
            {
                textEmbeddingOutputs[textEmbeddingSession.OutputNames[i]] = outputs[i];
            }

            return textEmbeddingOutputs;
        }


        public Image Run(string prompt, string negativePrompt, StableDiffusionConfig config)
        {

            //
            // 1. Create text embeddings for prompt with first tokenizer and text embedding model
            //
            var promptEmbeds = EncodePrompt(prompt, TokenizerSession, TextEmbeddingSession);

            //
            // 2. Create text embeddings for negative prompt with first tokenizer and text embedding model
            // If there is no negative prompt, we encode an empty string
            //

            //var negativePromptEmbeds = EncodePrompt(negativePrompt, TokenizerSession, TextEmbeddingSession);

            //
            // 3. Create text embeddings for the prompt with the second tokenizer and text embedding model
            // 
            var promptEmbeds2 = EncodePrompt(prompt, Tokenizer2Session, TextEmbedding2Session);

            //
            // 4. Create text embeddings for the negative prompt with the second tokenizer and text embedding model
            //
            //var negativePromptEmbeds2 = EncodePrompt(negativePrompt, Tokenizer2Session, TextEmbedding2Session);

            //
            // 5. Combine the embedding outputs into inputs to the UNet model
            //    - Concatenate the prompt embeddings from the two models
            //    - Use the text embeddings from the second model
            //    - Calculate the time embeds

            var lastHiddenState = promptEmbeds["last_hidden_state"];
            var lastHiddenStateShape = lastHiddenState.GetTensorTypeAndShape().Shape;
            var lastHiddenState2 = promptEmbeds2["last_hidden_state"];
            var lastHiddenState2Shape = lastHiddenState2.GetTensorTypeAndShape().Shape;

            var lastHiddenStateArray = lastHiddenState.GetTensorDataAsSpan<float>().ToArray();
            var lastHiddenState2Array = lastHiddenState2.GetTensorDataAsSpan<float>().ToArray();

            var buffer = new float[lastHiddenStateArray.Length + lastHiddenState2Array.Length];
            lastHiddenStateArray.CopyTo(buffer, 0);
            lastHiddenState2Array.CopyTo(buffer, lastHiddenStateArray.Length);

            long[] combinedHiddenStateShape = {1, lastHiddenStateShape[1], lastHiddenStateShape[2] + lastHiddenState2Shape[2]};

            var combinedHiddenState = OrtValue.CreateTensorValueFromMemory<float>(buffer, combinedHiddenStateShape);

            // Calculate time embeds
            float[] timeEmbedsBuffer = {config.Height, config.Width, 0, 0, config.Height, config.Width};
            long[] timeEmbedsShape = { 1, 6 };
            var timeEmbeds = OrtValue.CreateTensorValueFromMemory<float>(timeEmbedsBuffer, timeEmbedsShape);

            var unetInputs = new Dictionary<string, object>();
            unetInputs["hidden_states"] = combinedHiddenState;
            unetInputs["text_embeds"] = promptEmbeds2["text_embeds"];
            unetInputs["time_embeds"] = timeEmbeds;

            //
            // 6. Run the Unet model in a loop
            //
            long[] latentShape = { 1, 4, config.Width / 4, config.Height / 8 };
            double[] randomNormal = new double[config.Width * config.Height/2];

            double mean = 100;
            double stdDev = 10;

            Normal normalDist = new Normal(mean, stdDev);
            for (int i = 0; i < randomNormal.Length; i++)
            {
                randomNormal[i] = normalDist.Sample();
            }

            var latents = OrtValue.CreateTensorValueFromMemory<double>(randomNormal, latentShape);

            










            //var poolerOutput = textEmbeddingOutputs["pooler_output"];
            //var hiddenStates11 = textEmbeddingOutputs["hidden_state.11"];








            return null;
        }
    }
}
