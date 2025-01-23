using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ConsoleApp1
{
    public class BookShopChatClient : IChatClient
    {

        private readonly Dictionary<string, string> _keywordResponses = new()
        {
            ["children"] = "child movies like toy story and etc",
            ["horror"] = "do you enjoy horror, we got plenty of that",
        };

        private const string FallbackResponse =
            "I'm not quite sure about that. Could you clarify?";

        private readonly Dictionary<string, float[]> _keywordVectors = new()
        {
            //gatheres from running vectorize app, couild in theroy add as many as you want but for this experiment we only using 2
            ["horror"] = new float[] { 0.19245009124279022f, 0.19245009124279022f, 0.5773502588272095f, 0.5773502588272095f, 0.19245009124279022f, 0.19245009124279022f, 0.19245009124279022f, 0.19245009124279022f, 0.19245009124279022f, 0.19245009124279022f, 0.19245009124279022f, 0.7071067690849304f, 0.7071067690849304f },
            ["children"] = new float[] { 0.2182178795337677f, 0.2182178795337677f, 0.2182178795337677f, 0.4364357590675354f, 0.2182178795337677f, 0.2182178795337677f, 0.2182178795337677f, 0.4364357590675354f, 0.2182178795337677f, 0.2182178795337677f, 0.2182178795337677f, 0.2182178795337677f, 0.2182178795337677f, 0.2182178795337677f, 0.2182178795337677f, 0.7071067690849304f, 0.7071067690849304f }
        };

        public ChatClientMetadata Metadata { get; }

        public BookShopChatClient(Uri endpoint, string modelId)
        {
            Metadata = new("BookShopChatClient", endpoint, modelId);
        }

        public async Task<ChatCompletion> CompleteAsync(
            IList<ChatMessage> chatMessages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            await Task.Delay(100, cancellationToken);

            string userMessage = chatMessages
                .LastOrDefault(m => m.Role == ChatRole.User)?
                .Text ?? string.Empty;

            string response = await GetClerkResponseAsync(userMessage);

            return new ChatCompletion(new[]
            {
                new ChatMessage
                {
                    Role = ChatRole.Assistant,
                    Text = response
                }
            });
        }

        public async IAsyncEnumerable<StreamingChatCompletionUpdate> CompleteStreamingAsync(
            IList<ChatMessage> chatMessages,
            ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.Delay(100, cancellationToken);

            string userMessage = chatMessages
                .LastOrDefault(m => m.Role == ChatRole.User)?
                .Text ?? string.Empty;

            string response = await GetClerkResponseAsync(userMessage);

            string[] words = response.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            foreach (string word in words)
            {
                await Task.Delay(50, cancellationToken);
                yield return new StreamingChatCompletionUpdate
                {
                    Role = ChatRole.Assistant,
                    Text = word + " "
                };
            }
        }

        private async Task<string> GetClerkResponseAsync(string userInput)
        {
            float[] userEmbedding = await EmbedTextAsync(userInput);

            float bestSimilarity = -1.0f;
            string? bestKeyword = null;

            foreach (var (keyword, vector) in _keywordVectors)
            {
                float similarity = CosineSimilarity(userEmbedding, vector);
                if (similarity > bestSimilarity)
                {
                    bestSimilarity = similarity;
                    bestKeyword = keyword;
                }
            }

            float threshold = 0.5f; // Adjust as needed
            if (bestKeyword != null && bestSimilarity >= threshold)
            {
                return _keywordResponses[bestKeyword];
            }
            else
            {
                return FallbackResponse;
            }
        }

        private Task<float[]> EmbedTextAsync(string text)
        {
            var mlContext = new MLContext();

            var data = new List<InputData> { new InputData { Text = text } };

            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var pipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                options: new Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options
                {
                    WordFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options
                    {
                        NgramLength = 1, // Use unigrams (adjustable)
                        UseAllLengths = true
                    },
                    CharFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options
                    {
                        NgramLength = 1, // Use unigrams (adjustable)
                        UseAllLengths = true
                    }, 
                },
                inputColumnNames: nameof(InputData.Text));

            var model = pipeline.Fit(dataView);

            var transformedData = model.Transform(dataView);

            var features = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, reuseRowObject: false).FirstOrDefault();

            if (features != null && features.Features != null)
            {
                return Task.FromResult(features.Features);
            }
            else
            {
                return Task.FromResult(new float[0]);
            }
        }


        private static float CosineSimilarity(float[] a, float[] b)
        {
            int maxLength = Math.Max(a.Length, b.Length);

            // Pad the shorter vector with zeros
            float[] aAdjusted = a.Length < maxLength
                ? a.Concat(new float[maxLength - a.Length]).ToArray()
                : a;

            float[] bAdjusted = b.Length < maxLength
                ? b.Concat(new float[maxLength - b.Length]).ToArray()
                : b;

            float dot = 0f;
            float magA = 0f;
            float magB = 0f;

            for (int i = 0; i < maxLength; i++)
            {
                dot += aAdjusted[i] * bAdjusted[i];
                magA += aAdjusted[i] * aAdjusted[i];
                magB += bAdjusted[i] * bAdjusted[i];
            }

            float magnitudeProduct = (float)(Math.Sqrt(magA) * Math.Sqrt(magB));
            return magnitudeProduct == 0 ? 0 : dot / magnitudeProduct;
        }


        public TService? GetService<TService>(object? key = null) where TService : class => this as TService;
        public object? GetService(Type serviceType, object? serviceKey = null) => throw new NotImplementedException();
        void IDisposable.Dispose() { }
    }
    public class InputData
    {
        public string Text { get; set; }
    }

    public class TransformedData
    {
        [VectorType]
        public float[] Features { get; set; }
    }
}
