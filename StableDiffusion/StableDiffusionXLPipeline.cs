using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace StableDiffusion
{
    internal class StableDiffusionXLPipeline
    {
        OrtValue placeholder;

        InferenceSession TokenizerSession;
        Dictionary<string, OrtValue> TokenizerInputs;
        Dictionary<string, OrtValue> TokenizerOutputs;
        
        InferenceSession Tokenizer2Session;
        Dictionary<string, OrtValue> Tokenizer2Inputs;
        Dictionary<string, OrtValue> Tokenizer2Outputs;

        InferenceSession TextEmbeddingSession;
        Dictionary<string, OrtValue> TextEmbeddingInputs;
        Dictionary<string, OrtValue> TextEmbeddingOutputs;
       
        InferenceSession TextEmbedding2Session;
        Dictionary<string, OrtValue> TextEmbedding2Inputs;
        Dictionary<string, OrtValue> TextEmbedding2Outputs;

        InferenceSession UNetSession;
        Dictionary<string, OrtValue> UNetInputs;
        Dictionary<string, OrtValue> UNetOutputs;

        InferenceSession VaeDecoderSession;
        Dictionary<string, OrtValue> VaeDecoderInputs;
        Dictionary<string, OrtValue> VaeDecoderOutputs;


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
            var outputs = tokenizerSession.Run(runOptions, TokenizerInputs, tokenizerSession.OutputNames);

            // Set the outputs
            var tokenizerOutputs = new Dictionary<string, OrtValue>();
            for (int i = 0; i < outputs.Count; i++)
            {
                tokenizerOutputs[tokenizerSession.OutputNames[i]] = outputs[i];
            }
            var inputIds = TokenizerOutputs["input_ids"];
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
            TextEmbeddingInputs["input_ids"] = inputIds;

            // Run the text embedding session
            outputs = textEmbeddingSession.Run(runOptions, TextEmbeddingInputs, textEmbeddingSession.OutputNames);

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

            var lastHiddenState = promptEmbeds["last_hidden_state"];
            var lastHiddenStateShape = lastHiddenState.GetTensorTypeAndShape().Shape;
            var lastHiddenState2 = promptEmbeds2["last_hidden_state"];
            var lastHiddenState2Shape = lastHiddenState2.GetTensorTypeAndShape().Shape;

            // Last hidden state shape is {batch, sequence length, 1280}
            // UNet expects 

            var buffer = new float[1 * lastHiddenStateShape[1] * lastHiddenStateShape[2] + lastHiddenState2Shape[2]];

            // Copy the hidden state into the new buffer


            long[] unetInputShape = { 1, lastHiddenState2Shape[1], lastHiddenStateShape[2] + lastHiddenState2Shape[2] };

            // Creates an OrtValue from a buffer
            var unetInput = OrtValue.CreateTensorValueFromMemory<float>(buffer, unetInputShape);

            //var poolerOutput = textEmbeddingOutputs["pooler_output"];
            //var hiddenStates11 = textEmbeddingOutputs["hidden_state.11"];



            //UNetInputs["hidden_states"] = ;
            //UNetInputs["text_embeds"] = ;


            // 
            // 6. Create the time embeddings
            // 
            //    return new Tensor(
            //'float32',
            // Float32Array.from([height, width, 0, 0, height, width]),
            // [1, 6],
            //)




            // Notes:

            // Create a new buffer of a fixed type and length
            //var buffer = new int[data.Length];


            //var length = 2048;
            //var buffer = new float[length];

            // Create a span over the buffer
            // 0-1023
            // 1024-2047
            // Copy to the span over the buffer.

            //data1.CopyTo(buffer, dimension?);

            //long[] shape2 = { 2, 1024 };

            // Creates an OrtValue from a buffer
            //var newInput = OrtValue.CreateTensorValueFromMemory(buffer, shape2);

            // Span<float> array1;
            // Span<float> array2;
            // Allocate a third buffer
            // Either array copy or span copy array1 + array2 into array3

            // Span.CopyTo(another span)





            return null;
        }
    }
}
