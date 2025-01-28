using FaceONNX;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetection;
using ObjectDetection.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;
using System.Threading.Tasks;

namespace TinyYoloWebApp.Controllers
{
    public class FacesController : Controller
    {
        static FaceDetector faceDetector;
        static Face68LandmarksExtractor _faceLandmarksExtractor;
        static FaceEmbedder _faceEmbedder;


        public FacesController()
        {
            faceDetector = new FaceDetector();
            _faceLandmarksExtractor = new Face68LandmarksExtractor();
            _faceEmbedder = new FaceEmbedder();
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> TrainFaceImage(IFormFile UploadedImage)
        {
            if (UploadedImage == null || UploadedImage.Length == 0)
            {
                return BadRequest("No image file provided.");
            }

            var fitDirectory = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "fit");
            Directory.CreateDirectory(fitDirectory);

            var filePath = Path.Combine(fitDirectory, UploadedImage.FileName);
            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await UploadedImage.CopyToAsync(stream);
            }

            return View("Index");
        }

        [HttpPost]
        public async Task<IActionResult> ProcessFaceImage(IFormFile UploadedImage)
        {
            if (UploadedImage == null || UploadedImage.Length == 0)
            {
                return BadRequest("No image file provided.");
            }

            var scoreDirectory = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "score");
            Directory.CreateDirectory(scoreDirectory);

            var filePath = Path.Combine(scoreDirectory, UploadedImage.FileName);
            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await UploadedImage.CopyToAsync(stream);
            }

            using var theImage = Image.Load<Rgb24>(filePath);

            var embeddings = new Embeddings();
            var fitDirectory = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "fit");
            var fits = Directory.GetFiles(fitDirectory);


            float[] embedding;
            foreach (var fit in fits)
            {
                using var trainingImage = Image.Load<Rgb24>(fit);
                embedding = GetEmbedding(trainingImage);
                var name = Path.GetFileNameWithoutExtension(fit);
                embeddings.Add(embedding, name);
            }

            embedding = GetEmbedding(theImage);
            var proto = embeddings.FromSimilarity(embedding);
            var label = proto.Item1;
            var similarity = proto.Item2;

            // Clean up resources
            faceDetector.Dispose();
            _faceLandmarksExtractor.Dispose();
            _faceEmbedder.Dispose();
            ViewBag.ImageDetails = $"Image: {UploadedImage.FileName}, Classified as: {label}, Similarity: {similarity:F2}";

            return View("Index");
        }


        static float[] GetEmbedding(Image<Rgb24> image)
        {
            var array = GetImageFloatArray(image);
            var rectangles = faceDetector.Forward(array);
            var rectangle = rectangles.FirstOrDefault().Box;

            if (!rectangle.IsEmpty)
            {
                // landmarks
                var points = _faceLandmarksExtractor.Forward(array, rectangle);
                var angle = points.RotationAngle;

                // alignment
                var aligned = FaceProcessingExtensions.Align(array, rectangle, angle);
                return _faceEmbedder.Forward(aligned);
            }

            return new float[512];
        }

        static float[][,] GetImageFloatArray(Image<Rgb24> image)
        {
            var array = new[]
            {
                new float [image.Height,image.Width],
                new float [image.Height,image.Width],
                new float [image.Height,image.Width]
            };

            image.ProcessPixelRows(pixelAccessor =>
            {
                for (var y = 0; y < pixelAccessor.Height; y++)
                {
                    var row = pixelAccessor.GetRowSpan(y);
                    for (var x = 0; x < pixelAccessor.Width; x++)
                    {
                        array[2][y, x] = row[x].R / 255.0F;
                        array[1][y, x] = row[x].G / 255.0F;
                        array[0][y, x] = row[x].B / 255.0F;
                    }
                }
            });

            return array;
        }
    }
}
