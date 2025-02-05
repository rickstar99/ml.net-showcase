using System;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace SentimentAnalysis
{
    internal class Program
    {
        private static readonly HttpClient httpClient = new HttpClient();

        static async Task Main(string[] args)
        {
            // Set a custom User-Agent header (required by Reddit API)
            httpClient.DefaultRequestHeaders.Add("User-Agent", "SentimentAnalysisApp/1.0");

            while (true)
            {
                Console.Write("\nEnter search query (or type 'exit' to quit): ");
                string input = Console.ReadLine();

                if (input?.ToLower() == "exit")
                {
                    Console.WriteLine("Goodbye!");
                    break;
                }

                // Build the Reddit API URL
                string apiUrl = $"https://www.reddit.com/r/all/search.json?q={Uri.EscapeDataString(input)}&sort=relevance&limit=10";

                try
                {
                    // Fetch data from Reddit API
                    var response = await httpClient.GetAsync(apiUrl);
                    response.EnsureSuccessStatusCode(); // Throw if HTTP request fails

                    // Parse the JSON response
                    var jsonResponse = await response.Content.ReadAsStringAsync();
                    var redditData = JsonSerializer.Deserialize<RedditResponse>(jsonResponse);

                    if (redditData?.Data?.Children == null || !redditData.Data.Children.Any())
                    {
                        Console.WriteLine("No results found.");
                        continue;
                    }

                    // Aggregate all post titles and comments into a single string
                    StringBuilder combinedText = new StringBuilder();

                    foreach (var post in redditData.Data.Children)
                    {
                        combinedText.AppendLine(post.Data.Title);

                        var commentsUrl = $"https://www.reddit.com{post.Data.Permalink}.json";
                        var commentsResponse = await httpClient.GetAsync(commentsUrl);
                        commentsResponse.EnsureSuccessStatusCode();

                        var commentsJsonResponse = await commentsResponse.Content.ReadAsStringAsync();
                        var commentsData = JsonSerializer.Deserialize<RedditCommentResponse[]>(commentsJsonResponse);

                        if (commentsData != null && commentsData.Length > 1)
                        {
                            var comments = commentsData[1].Data.Children;
                            foreach (var comment in comments)
                            {
                                // Add the comment body
                                combinedText.AppendLine(comment.Data.Body);
                            }
                        }
                    }

                    // Perform sentiment analysis on the combined text
                    var sampleData = new SentimentAnalysis.ModelInput()
                    {
                        Comment = combinedText.ToString()
                    };

                    // Load model and predict output
                    var result = SentimentAnalysis.Predict(sampleData);
                    string overallSentiment = ((int)result.PredictedLabel) == 0 ? "Good vibes" : "Bad Vibes";

                    // Output the overall sentiment
                    Console.WriteLine($"Overall sentiment for the search query '{input}': {overallSentiment}");
                }
                catch (HttpRequestException ex)
                {
                    Console.WriteLine($"Error fetching data from Reddit API: {ex.Message}");
                }
                catch (JsonException ex)
                {
                    Console.WriteLine($"Error parsing Reddit API response: {ex.Message}");
                }
            }
        }
    }

    // Classes to model the Reddit API JSON response for posts
    public class RedditResponse
    {
        [JsonPropertyName("data")]
        public RedditData Data { get; set; }
    }

    public class RedditData
    {
        [JsonPropertyName("children")]
        public RedditPost[] Children { get; set; }
    }

    public class RedditPost
    {
        [JsonPropertyName("data")]
        public PostData Data { get; set; }
    }

    public class PostData
    {
        [JsonPropertyName("title")]
        public string Title { get; set; }

        [JsonPropertyName("permalink")]
        public string Permalink { get; set; }
    }

    // Classes to model the Reddit API JSON response for comments
    public class RedditCommentResponse
    {
        [JsonPropertyName("data")]
        public CommentData Data { get; set; }
    }

    public class CommentData
    {
        [JsonPropertyName("children")]
        public Comment[] Children { get; set; }
    }

    public class Comment
    {
        [JsonPropertyName("data")]
        public CommentBody Data { get; set; }
    }

    public class CommentBody
    {
        [JsonPropertyName("body")]
        public string Body { get; set; }
    }
}