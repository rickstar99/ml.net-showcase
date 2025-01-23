using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace StringVectorizer
{
    class Program
    {
        static void Main(string[] args)
        {
            while(true) {
            Console.WriteLine("Enter a string to vectorize:");
            string input = Console.ReadLine();

            int vectorLength = 1024;
            double[] vector = VectorizeStringWithMLNET(input, vectorLength);

            Console.WriteLine("Vectorized Output (Length = " + vector.Length + "):");
            Console.WriteLine(string.Join("f, ", vector));
                }
        }

        static double[] VectorizeStringWithMLNET(string input, int vectorLength)
        {
            // Create an ML.NET context
            MLContext mlContext = new MLContext();

            // Define input data schema
            var inputData = new[] { new InputData { Text = input } };

            // Load input data into an IDataView
            var dataView = mlContext.Data.LoadFromEnumerable(inputData);

            // Define a pipeline that converts text into a fixed-size vector
            var pipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnNames: "Text",
                options: new Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options
                {
                    WordFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options
                    {
                        NgramLength = 1, // Use unigrams (adjustable)
                    },
                    CharFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options
                    {
                        NgramLength = 1, // Use unigrams (adjustable)
                    }, // Disable character n-grams
                });

            // Fit the pipeline to the data
            var transformer = pipeline.Fit(dataView);

            // Transform the input data
            var transformedData = transformer.Transform(dataView);

            // Extract the feature vector
            var featuresColumn = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, reuseRowObject: false);
            var featureVector = featuresColumn.First().Features;

            return featureVector.Select(f => (double)f).ToArray();
        }


        // Input data class
        public class InputData
        {
            public string Text { get; set; }
        }

        // Transformed data class
        public class TransformedData
        {
            public float[] Features { get; set; }
        }
    }
}
