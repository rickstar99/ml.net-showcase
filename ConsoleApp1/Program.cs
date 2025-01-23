using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace ConsoleApp1
{
    public static class Program
    {
        public static async Task Main()
        {
            Uri endpoint = new("http://my-bookshop.ai");
            IChatClient bot = new BookShopChatClient(endpoint, "demo-book-model");

            Console.WriteLine("Welcome to the Book Shop Clerk chatbot with vector-based matching!");
            Console.WriteLine("Type your questions, or 'exit' to quit.\n");

            while (true)
            {
                Console.Write("You: ");
                string? input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input) || input.ToLower() == "exit")
                {
                    Console.WriteLine("Bot: Thanks for chatting! Bye.");
                    break;
                }

                var messages = new List<ChatMessage>
                {
                    new ChatMessage { Role = ChatRole.User, Text = input }
                };

                // Non-streaming response
                ChatCompletion completion = await bot.CompleteAsync(messages);

                // Print the first assistant message (assuming one)
                if (completion.Message != null)
                {
                    Console.WriteLine($"Bot: {completion.Message.Text}\n");
                }
                else
                {
                    Console.WriteLine("Bot: (No response?)\n");
                }
            }
        }
    }
}
