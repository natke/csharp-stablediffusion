using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace StableDiffusion.ML.OnnxRuntime
{
    public static class TextProcessing2
    {
        public static DenseTensor<float> PreprocessText(String prompt, StableDiffusionConfig config)
        {
            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TokenizeText(prompt, config);
            var textPromptEmbeddings = TextEncoder(textTokenized, config);

            // Create uncond_input of blank tokens
            var uncondInputTokens = CreateUncondInput();
            var uncondEmbedding = TextEncoder(uncondInputTokens, config);

            // Concant textEmeddings and uncondEmbedding
            DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 2048 });

            // TODO Figure out how to concatenate OrtValues
            for (var i = 0; i < textPromptEmbeddings.GetTensorDataAsSpan<long>().Length; i++)
            {
                textEmbeddings[0, i / 2048, i % 2048] = uncondEmbedding.GetTensorDataAsSpan<long>()[i];
                textEmbeddings[1, i / 2048, i % 2048] = textPromptEmbeddings.GetTensorDataAsSpan<long>()[i];
            }
            return textEmbeddings;
        }
        public static OrtValue TokenizeText(string text, StableDiffusionConfig config)
        {
            long[] shape = { 1 };

            using var inputTensor = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);

            inputTensor.StringTensorSetElementAt(text.AsSpan(), 0);

            // Create session options for custom op of extensions
            var sessionOptions = new SessionOptions();
            sessionOptions.RegisterOrtExtensions();
            
            // Create an InferenceSession from the onnx clip tokenizer.
            var tokenizeSession = new InferenceSession(config.TokenizerOnnxPath, sessionOptions);

            var input = new Dictionary<string, OrtValue> {
                { "string_input",  inputTensor }
            };


            using var runOptions = new RunOptions();


            // Run session and send the input data in to get inference output. 
            var outputs = tokenizeSession.Run(runOptions, input, tokenizeSession.OutputNames);

            // TODO Figure out how to pad OrtValues
            //var modelMaxLength = 77;
            //var paddedInputs = new List<NamedOnnxValue>();
            // Pad array with 49407 until length is modelMaxLength
            //if (inputIds.Length < modelMaxLength)
            //{
            //    var pad = Enumerable.Repeat(49407, 77 - inputIds.Length).ToArray();
                //paddedInputs = inputIds.Concat(pad).ToArray();
            //}

            return outputs[0];

        }

        public static OrtValue CreateUncondInput()
        {
            // Create an array of empty tokens for the unconditional input.
            var blankTokenValue = 49407;
            var modelMaxLength = 77;
            long[] inputIds = new long[modelMaxLength];
            //inputIds.Add(49406); // TODO What is this?
            
            for (var i = 0; i < modelMaxLength; i++)
            {
                inputIds[i] = blankTokenValue;
            }
            
            long[] inputIdsShape = { modelMaxLength };

            return OrtValue.CreateTensorValueFromMemory(inputIds, inputIdsShape);

            
        }

        public static OrtValue TextEncoder(OrtValue inputIds, StableDiffusionConfig config)
        {
            var input = new Dictionary<string, OrtValue> {
                { "input_ids",  inputIds }
            };

            Console.WriteLine(inputIds.GetTensorTypeAndShape());

            // Set CUDA EP
            var sessionOptions = config.GetSessionOptionsForEp();
            var encodeSession = new InferenceSession(config.TextEncoder2OnnxPath, sessionOptions);


            // Run inference.
            using var runOptions = new RunOptions();

            var outputs = encodeSession.Run(runOptions, input, encodeSession.OutputNames);

            //var lastHiddenState = (encoded.ToList().First().Value as IEnumerable<float>).ToArray();
            //var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

            //return lastHiddenStateTensor;
            return outputs[0];

        }

    }
}