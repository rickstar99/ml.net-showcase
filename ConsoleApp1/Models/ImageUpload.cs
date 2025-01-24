using Microsoft.AspNetCore.Http;

namespace ObjectDetection.Models
{
    public class ImageUpload
    {
        public IFormFile UploadedImage { get; set; }
    }
}
