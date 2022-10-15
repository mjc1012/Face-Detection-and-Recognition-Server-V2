using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Face_Detection_and_Recognition_Server_V2.Helper_Classes
{
    public class Config
    {
        public static string augmentedDataDirectory = @"C:\Users\tsg\Documents\Face Recongition Thesis\thesis version 2\Augmented Face Dataset (Colored)\";
        public static string augmentedDataDirectoryGrayscale = @"C:\Users\tsg\Documents\Face Recongition Thesis\thesis version 2\Augmented Face Dataset (Grayscale)\";
        public static string augmentedDataInceptionDirectory = @"C:\Users\tsg\Documents\Face Recongition Thesis\thesis version 2\Augmentated Face Dataset (Colored and for Inception)\";
        public static string augmentedDataInceptionDirectoryGrayscale = @"C:\Users\tsg\Documents\Face Recongition Thesis\thesis version 2\Augmentated Face Dataset (Grayscale and for Inception)\";
        public static string dataDirectory = @"C:\Users\tsg\Documents\Face Recongition Thesis\thesis version 2\Face Dataset\";
        public static string projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
        public static string tensorflowFaceRecognitionModelsDirectory = Path.Combine(projectDirectory, "Tensorflow Face Recognition Models");
        public static string serverDotNetFaceRecognitionModelsDirectory = Path.Combine(projectDirectory, "Dotnet Face Recognition Models");
        public static string clientFaceRecongitionModelsDirectory = @"C:\Users\tsg\Documents\Face Recongition Thesis\thesis version 2\Face-Detection-and-Recognition-Client\Face Recognition Models\";
    }
}