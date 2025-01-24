using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetection.Models;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;

namespace TinyYoloWebApp.Controllers
{
    public class HomeController : Controller
    {
        private readonly string _modelPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "models", "TinyYolo2_model.onnx");
        private readonly MLContext _mlContext;

        public HomeController()
        {
            _mlContext = new MLContext();
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Index(IFormFile UploadedImage)
        {
            if (UploadedImage != null && UploadedImage.Length > 0)
            {
                var uploadsDir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads");
                Directory.CreateDirectory(uploadsDir); // Ensure uploads directory exists

                var filePath = Path.Combine(uploadsDir, UploadedImage.FileName);

                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await UploadedImage.CopyToAsync(stream);
                }

                // Run ONNX model and process image
                var processedImagePath = RunOnnxModel(filePath);

                ViewBag.ProcessedImagePath = $"/uploads/{Path.GetFileName(processedImagePath)}";
            }

            return View("Index");
        }

        private string RunOnnxModel(string imagePath)
        {
            // Load the ONNX model and pipeline

            var dataView = _mlContext.Data.LoadFromEnumerable(new[] { new { ImagePath = imagePath } });

            var pipeline = _mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: "ImagePath")
                .Append(_mlContext.Transforms.ResizeImages("image", imageWidth: 416, imageHeight: 416, inputColumnName: "image"))
                .Append(_mlContext.Transforms.ExtractPixels("image"))
                .Append(_mlContext.Transforms.ApplyOnnxModel(modelFile: _modelPath, outputColumnNames: new[] { "grid" }, inputColumnNames: new[] { "image" }));


            var model = pipeline.Fit(dataView);
            var scoredData = model.Transform(dataView);

            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>("grid");
            YoloOutputParser parser = new YoloOutputParser();

            var boundingBoxes = probabilities.Select(probability => parser.ParseOutputs(probability)).Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

            // Draw bounding boxes on the image
            string processedImagePath = DrawBoundingBox(imagePath, boundingBoxes.First());

            return processedImagePath;
        }


     private string DrawBoundingBox(string imagePath, List<YoloBoundingBox> boundingBoxes)
        {
            string uploadsDir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads");
            Directory.CreateDirectory(uploadsDir); // Ensure the uploads folder exists

            // Load the original image
            using var image = Image.FromFile(imagePath);
            var originalImageWidth = image.Width;
            var originalImageHeight = image.Height;

            using var graphics = Graphics.FromImage(image);
            graphics.CompositingQuality = CompositingQuality.HighQuality;
            graphics.SmoothingMode = SmoothingMode.HighQuality;
            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

            var font = new Font("Arial", 12, FontStyle.Bold);
            var brush = new SolidBrush(Color.Yellow);
            var pen = new Pen(Color.Red, 3);

            // Draw each bounding box
            foreach (var box in boundingBoxes)
            {
                // Scale the bounding box to the original image size
                var x = originalImageWidth * box.Dimensions.X / 416; // YOLO model size is 416x416
                var y = originalImageHeight * box.Dimensions.Y / 416;
                var width = originalImageWidth * box.Dimensions.Width / 416;
                var height = originalImageHeight * box.Dimensions.Height / 416;

                // Draw the rectangle
                graphics.DrawRectangle(pen, x, y, width, height);

                // Draw the label and confidence
                string label = $"{box.Label} ({box.Confidence:P1})";
                var textSize = graphics.MeasureString(label, font);
                graphics.FillRectangle(new SolidBrush(Color.FromArgb(125, Color.Black)), x, y - textSize.Height, textSize.Width, textSize.Height);
                graphics.DrawString(label, font, brush, x, y - textSize.Height);
            }

            // Save the processed image with a unique name
            string processedImageName = $"processed_{Path.GetFileName(imagePath)}";
            string processedImagePath = Path.Combine(uploadsDir, processedImageName);
            image.Save(processedImagePath, ImageFormat.Jpeg);

            return processedImagePath;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
            public const float mean = 117;
            public const bool channelsLast = true;
        }

        public class YoloInput
        {
            public string ImagePath { get; set; }
        }

        public class YoloPrediction
        {
            [ColumnName("grid")]
            public float[] PredictedLabels { get; set; }
        }
    }
}
