using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetection;
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
        public async Task<IActionResult> ProcessLuffy(IFormFile UploadedImage)
        {
            if (UploadedImage != null && UploadedImage.Length > 0)
            {
                var uploadsDir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads");
                Directory.CreateDirectory(uploadsDir); // Ensure uploads directory exists

                var filePath = Path.Combine(uploadsDir, UploadedImage.FileName);

                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await UploadedImage.CopyToAsync(stream);
                    stream.Close();
                }

                var processedImagePath = RunLuffyModel(filePath);

                ViewBag.ProcessedImagePath = $"/processed/{Path.GetFileName(processedImagePath)}";
            }

            return View("Index");
        }


        [HttpPost]
        public async Task<IActionResult> ProcessTomCruise(IFormFile UploadedImage)
        {
            if (UploadedImage != null && UploadedImage.Length > 0)
            {
                var uploadsDir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads");
                Directory.CreateDirectory(uploadsDir); // Ensure uploads directory exists

                var filePath = Path.Combine(uploadsDir, UploadedImage.FileName);

                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await UploadedImage.CopyToAsync(stream);
                    stream.Close();
                }

                var processedImagePath = RunFacesModel(filePath);


                ViewBag.ProcessedImagePath = $"/processed/{Path.GetFileName(processedImagePath)}";
            }

            return View("Index");
        }

        private string RunFacesModel(string imagePath)
        {
            // Create the directory if it doesn't exist
            string uploadsDir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "processed");
            Directory.CreateDirectory(uploadsDir);

            // 1) Create the ML image and prediction input
            var image = MLImage.CreateFromFile(imagePath);
            FacesDetection.ModelInput sampleData = new FacesDetection.ModelInput
            {
                Image = image
            };

            var predictionResult = FacesDetection.Predict(sampleData);


            var boxesAnonymous = predictionResult.PredictedBoundingBoxes
                ?.Chunk(4)
                .Select(coords => new
                {
                    XTop = coords[0],
                    YTop = coords[1],
                    XBottom = coords[2],
                    YBottom = coords[3]
                })
                .Zip(predictionResult.Score, (box, score) => new { Box = box, Score = score })
                .ToList();

            var boxes = boxesAnonymous.Select(b => (
                    XTop: b.Box.XTop,
                    YTop: b.Box.YTop,
                    XBottom: b.Box.XBottom,
                    YBottom: b.Box.YBottom,
                    Score: b.Score
                )).Where(b => b.Score > 0.7).ToList();

            var processedImagePath = DrawBoundingBoxes(imagePath, boxes);

            return processedImagePath;
        }

        private string RunLuffyModel(string imagePath)
        {
            // Create the directory if it doesn't exist
            string uploadsDir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "processed");
            Directory.CreateDirectory(uploadsDir);

            // 1) Create the ML image and prediction input
            var image = MLImage.CreateFromFile(imagePath);
            LuffyDetection.ModelInput sampleData = new LuffyDetection.ModelInput
            {
                Image = image
            };

            var predictionResult = LuffyDetection.Predict(sampleData);


            var boxesAnonymous = predictionResult.PredictedBoundingBoxes
                ?.Chunk(4)
                .Select(coords => new
                {
                    XTop = coords[0],
                    YTop = coords[1],
                    XBottom = coords[2],
                    YBottom = coords[3]
                })
                .Zip(predictionResult.Score, (box, score) => new { Box = box, Score = score })
                .ToList();

            var boxes = boxesAnonymous.Select(b => (
                    XTop: b.Box.XTop,
                    YTop: b.Box.YTop,
                    XBottom: b.Box.XBottom,
                    YBottom: b.Box.YBottom,
                    Score: b.Score
                )).ToList();

            var processedImagePath = DrawBoundingBoxes(imagePath, boxes);
            
            return processedImagePath;
        }
        private string DrawBoundingBoxes( string imagePath, IEnumerable<(float XTop, float YTop, float XBottom, float YBottom, float Score)> boxes)
        {
            try
            {
                string processedDir = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "processed");
                Directory.CreateDirectory(processedDir);

                if (!System.IO.File.Exists(imagePath))
                {
                    throw new FileNotFoundException("Image file not found.", imagePath);
                }

                var fileInfo = new FileInfo(imagePath);
                if (fileInfo.Length == 0)
                {
                    throw new Exception($"Image file '{imagePath}' is 0 bytes. Possibly invalid or corrupted.");
                }

                using var ms = new MemoryStream(System.IO.File.ReadAllBytes(imagePath));
                using var image = Image.FromStream(ms);  // This can still throw if corrupted
                using var graphics = Graphics.FromImage(image);
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

                using var pen = new Pen(Color.Red, 3);
                using var font = new Font("Arial", 12, FontStyle.Bold);
                using var brush = new SolidBrush(Color.Yellow);

                foreach (var box in boxes)
                {
                    float x = box.XTop;
                    float y = box.YTop;
                    float width = box.XBottom - x;
                    float height = box.YBottom - y;

                    graphics.DrawRectangle(pen, x, y, width, height);

                    string scoreText = box.Score.ToString("0.00");
                    var textSize = graphics.MeasureString(scoreText, font);

                    graphics.FillRectangle(new SolidBrush(Color.FromArgb(125, Color.Black)),
                                           x, y - textSize.Height, textSize.Width, textSize.Height);

                    graphics.DrawString(scoreText, font, brush, x, y - textSize.Height);
                }

                string processedImageName = $"processed_{Path.GetFileName(imagePath)}";
                string processedImagePath = Path.Combine(processedDir, processedImageName);
                image.Save(processedImagePath, ImageFormat.Jpeg);

                return processedImagePath;
            } catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }
        }

    }
}
