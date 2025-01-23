namespace SentimentAnalysis
{
    internal class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                Console.Write("\nYou: ");
                string input = Console.ReadLine();

                if (input?.ToLower() == "exit")
                {
                    Console.WriteLine("Goodbye!");
                    break;
                }
                var sampleData = new SentimentAnalysis.ModelInput()
                {
                    Comment = input
                };

                //Load model and predict output
                var result = SentimentAnalysis.Predict(sampleData);
                string prediction = ((int)result.PredictedLabel) == 0 ? "friendly" : "toxic";
                Console.WriteLine($"Analyser: {result.Comment} <---> {prediction}" );
            }
        }
    }
}
