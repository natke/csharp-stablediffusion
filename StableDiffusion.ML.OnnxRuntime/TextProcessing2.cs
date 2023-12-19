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
            IEnumerable<OrtValue> textTokenized = TokenizeText(prompt, config);
            OrtValue textPromptEmbeddings = TextEncoder(textTokenized, config)["text_embeds"];
            var textPromptEmbeddingsSpan = textPromptEmbeddings.GetTensorDataAsSpan<float>();
            var textPromptEmbeddingsLength = textPromptEmbeddingsSpan.Length;

            // Create uncond_input of blank tokens
            IEnumerable<OrtValue> uncondInputTokens = CreateUncondInput();
            OrtValue uncondEmbeddings = TextEncoder(uncondInputTokens, config)["text_embeds"];
            var uncondEmbeddingsSpan = uncondEmbeddings.GetTensorDataAsSpan<float>();
            var uncondEmbeddingsLength = uncondEmbeddingsSpan.Length;

            // Concat textEmeddings and uncondEmbedding
            DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 1, uncondEmbeddingsLength + textPromptEmbeddingsLength});

            for (var i = 0; i < uncondEmbeddingsLength; i++)
            {
                textEmbeddings[0, i] = uncondEmbeddingsSpan[i];
            }

            for (var i = uncondEmbeddingsLength;  i < uncondEmbeddingsLength + textPromptEmbeddingsLength; i++)
            {
                textEmbeddings[0, i] = textPromptEmbeddingsSpan[i - uncondEmbeddingsLength];
            }
            return textEmbeddings;
        }
        public static IEnumerable<OrtValue> TokenizeText(string text, StableDiffusionConfig config)
        {
            long[] shape = { 2 };

            using var inputTensor = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);

            inputTensor.StringTensorSetElementAt(text.AsSpan(), 0);

            // Create session options for custom op of extensions
            var sessionOptions = new SessionOptions();
            sessionOptions.RegisterOrtExtensions();
            
            // Create an InferenceSession from the onnx clip tokenizer.
            var tokenizeSession = new InferenceSession(config.TokenizerOnnxPath, sessionOptions);

            // Run session
            var input = new Dictionary<string, OrtValue> {
                { "string_input",  inputTensor }
            };
            using var runOptions = new RunOptions();

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

            var output0 = outputs.ElementAt(0);
            var tensorTypeAndShape = output0.GetTensorTypeAndShape();

            return outputs;

        }

        public static IEnumerable<OrtValue> CreateUncondInput()
        {
            List<OrtValue> returnVal = new List<OrtValue>();
         
            // Create an array of empty tokens for the unconditional input.
            var blankTokenValue = 49407;
            var modelMaxLength = 77;
            long[] inputIds = new long[modelMaxLength];
            //inputIds.Add(49406); // TODO What is this?
            
            for (var i = 0; i < modelMaxLength; i++)
            {
                inputIds[i] = blankTokenValue;
            }
            
            long[] inputIdsShape = { 1, modelMaxLength };

            returnVal.Add(OrtValue.CreateTensorValueFromMemory(inputIds, inputIdsShape));

            return returnVal;
            
        }

        public static Dictionary<string, OrtValue> TextEncoder(IEnumerable<OrtValue> inputs, StableDiffusionConfig config)
        {
            var input = new Dictionary<string, OrtValue> {
                { "input_ids",  inputs.ElementAt(0) }
            };

            // Set CUDA EP
            var sessionOptions = config.GetSessionOptionsForEp();
            var encodeSession = new InferenceSession(config.TextEncoder2OnnxPath, sessionOptions);


            // Run inference.
            using var runOptions = new RunOptions();

            var outputs = encodeSession.Run(runOptions, input, encodeSession.OutputNames);

            var returnVal = new Dictionary<string, OrtValue> {
                { "text_embeds", outputs.ElementAt(0) },
                { "last_hiddenstate", outputs.ElementAt(1) }
            };


            //var lastHiddenState = (encoded.ToList().First().Value as IEnumerable<float>).ToArray();
            //var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

            //return lastHiddenStateTensor;
            return returnVal;

        }

    }
}