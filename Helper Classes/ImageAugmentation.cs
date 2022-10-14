using AForge.Imaging.ColorReduction;
using AForge.Imaging.Filters;
using AForge.Math.Random;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;
using Random = System.Random;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.tensorflow;
using AForge;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace Face_Detection_and_Recognition_Server_V2.Helper_Classes
{
    public static class ImageAugmentation
    {
        static int count = 0;

        public static void runImageAugmentation(int augmentationChoice)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            AdditionalMethods.ConsoleWriteHeader("*** Image Augmentation ***");

            string[] subDirectories = Directory.GetDirectories(Config.dataDirectory);
            foreach (string subDirectory in subDirectories)
            {
                count = 0;
                var directory = new DirectoryInfo(subDirectory);
                FileInfo[] files = directory.GetFiles();
                foreach (FileInfo file in files)
                {
                    Bitmap image = new Bitmap(directory + "\\" + file.Name);
                    image = sharpenImage(image);
                    if (augmentationChoice == 1)
                    {
                        image = toGrayscale(image);
                        image = histogramEqualizationGrayscale(image.ToMat());
                    }
                    else
                    {
                        image = histogramEqualizationColored(image);
                    }

                    var imageVariations4 = ImageAugmentation.imageFlip(image);
                    foreach (Bitmap v4 in imageVariations4){
                        var imageVariations1 = ImageAugmentation.imageRotate(v4);

                        foreach (Bitmap v1 in imageVariations1)
                        {
                            var imageVariations2 = (augmentationChoice == 1) ? ImageAugmentation.imageFiltersGrayscale(v1) : ImageAugmentation.imageFilters(v1);

                            Random random = new Random();
                            var randomNumbers = (augmentationChoice == 1)? Enumerable.Range(0, 4).OrderBy(x => random.Next()).Take(4).ToList() : Enumerable.Range(0, 5).OrderBy(x => random.Next()).Take(5).ToList();
                            for (int i = 0; i < ((augmentationChoice == 1) ? 4 : 5); i++)
                            {
                                ImageAugmentation.saveImage(imageVariations2[i], directory.Name, augmentationChoice);
                                Bitmap imageVariations3 = imageVariations2[i];
                                switch (randomNumbers[i])
                                {
                                    case 0:
                                        imageVariations3 = blurImage(imageVariations2[i]);
                                        break;
                                    case 1:
                                        imageVariations3 = randomCutout(imageVariations2[i].ToMat());
                                        break;
                                    case 2:
                                        imageVariations3 = additiveNoise(imageVariations2[i]);
                                        break;
                                    case 3:
                                        imageVariations3 = imageDistortion(imageVariations2[i]);
                                        break;
                                    case 4:
                                        imageVariations3 = saltAndPepper(imageVariations2[i]);
                                        break;
                                }
                                ImageAugmentation.saveImage(imageVariations3, directory.Name, augmentationChoice);
                            }
                        }
                    }
                }
            }

            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Image Augmentation took: " + (elapsedMs / 1000).ToString() + " seconds");
        }

        public static Bitmap histogramEqualizationGrayscale(Mat image)
        {
            Cv2.EqualizeHist(image, image);
            return image.ToBitmap();
        }

        public static Bitmap histogramEqualizationColored(Bitmap Image)
        {
            HistogramEqualization filter = new HistogramEqualization();
            filter.ApplyInPlace(Image);
            return Image;
        }

        public static Bitmap imageDistortion(Bitmap image)
        {
            WaterWave filter = new WaterWave();
            filter.HorizontalWavesCount = 2;
            filter.HorizontalWavesAmplitude = 2;
            filter.VerticalWavesCount = 2;
            filter.VerticalWavesAmplitude = 2;
            return filter.Apply(image);
        }

        public static Bitmap saltAndPepper(Bitmap image)
        {
            SaltAndPepperNoise filter = new SaltAndPepperNoise(10);
            filter.ApplyInPlace(image);

            return image;
        }

        public static Bitmap toGrayscale(Bitmap image)
        {
            Grayscale filter = new Grayscale(0.2125, 0.7154, 0.0721);
            return filter.Apply(image);
        }

        public static Bitmap rotateImage(Bitmap image, int angle)
        {
            RotateNearestNeighbor filter = new RotateNearestNeighbor(angle, true);
            return filter.Apply(image);
        }

        public static List<Bitmap> imageFlip(Bitmap image)
        {
            Random random = new Random();

            var imageVariations = new List<Bitmap>() {
            { (Bitmap)image.Clone() },
            { horizontalFlip((Bitmap)image.Clone()) },
            };
            return imageVariations;
        }

        public static List<Bitmap> imageRotate(Bitmap image)
        {
            Random random = new Random();

            var imageVariations = new List<Bitmap>() {
            { (Bitmap)image.Clone() },
            { rotateImage((Bitmap)image.Clone(),25) },
            { rotateImage((Bitmap)image.Clone(),-25) },
            };
            return imageVariations;
        }


        public static Bitmap changeBrightness(Bitmap image, int adjustValue)
        {
            Random random = new Random();
            if (random.Next(0, 2) == 1) adjustValue = -adjustValue;
            BrightnessCorrection brightnessCorrection = new BrightnessCorrection(adjustValue);
            return brightnessCorrection.Apply(image);
        }

        public static Bitmap changeContrast(Bitmap image, int adjustValue)
        {
            Random random = new Random();
            if (random.Next(0, 2) == 1) adjustValue = -adjustValue;
            ContrastCorrection contrastCorrection = new ContrastCorrection(adjustValue);
            return contrastCorrection.Apply(image);
        }

        public static Bitmap changeSaturation(Bitmap image, float adjustValue)
        {
            Random random = new Random();
            if (random.Next(0, 2) == 1) adjustValue = -adjustValue;
            SaturationCorrection saturationCorrection = new SaturationCorrection(adjustValue / 100);
            return saturationCorrection.Apply(image);
        }

        public static Bitmap changeGamma(Bitmap image, double adjustValue)
        {
            GammaCorrection gammaCorrection = new GammaCorrection(adjustValue / 100);
            gammaCorrection.ApplyInPlace(image);
            return image;
        }

        public static Bitmap sharpenImage(Bitmap image)
        {
            GaussianSharpen filter = new GaussianSharpen(4, 11);
            return filter.Apply(image);
        }
        public static List<Bitmap> imageFilters(Bitmap image)
        {
            Random random = new Random();
            var imageVariations = new List<Bitmap>() {
            { (Bitmap)image.Clone() },
            { changeBrightness((Bitmap)image.Clone(), random.Next(40, 65)) },
            { changeContrast((Bitmap)image.Clone(), random.Next(40, 65)) },
            { changeSaturation((Bitmap)image.Clone(), random.Next(20, 35)) },
            { changeGamma((Bitmap)image.Clone(), random.Next(30, 50)) }
            };
            return imageVariations;
        }

        public static List<Bitmap> imageFiltersGrayscale(Bitmap image)
        {
            Random random = new Random();
            var imageVariations = new List<Bitmap>() {
            { (Bitmap)image.Clone() },
            { changeBrightness((Bitmap)image.Clone(), random.Next(40, 65)) },
            { changeContrast((Bitmap)image.Clone(), random.Next(40, 65)) },
            { changeGamma((Bitmap)image.Clone(), random.Next(30, 60)) },
            };
            return imageVariations;
        }

        public static Bitmap randomCutout(Mat image)
        {
            Random rand = new Random();
            int x = rand.Next(10, 154);
            int y = rand.Next(10, 154);
            Cv2.Rectangle(image, new Point(x, y), new Point(x + 70, y + 70), Scalar.Black, -1);
            return image.ToBitmap();
        }

        public static Bitmap additiveNoise(Bitmap image)
        {
            Random random = new Random();
            IRandomNumberGenerator generator = new UniformGenerator(new AForge.Range(-50, 50));
            AdditiveNoise filter = new AdditiveNoise(generator);
            filter.ApplyInPlace(image);

            return image;
        }

        public static Bitmap blurImage(Bitmap image)
        {
            GaussianBlur filter = new GaussianBlur(4, 11);
            return filter.Apply(image);
        }

        public static Bitmap horizontalFlip(Bitmap image)
        {
            image.RotateFlip(RotateFlipType.RotateNoneFlipX);
            return image;
        }

        public static void saveImage(Bitmap image, string filename, int augmentationChoice)
        {

            Cv2.Resize(image.ToMat(), image.ToMat(), new Size(224, 224));

            if (augmentationChoice == 1)
            {
                if (!Directory.Exists(Config.augmentedDataDirectoryGrayscale + filename)) Directory.CreateDirectory(Config.augmentedDataDirectoryGrayscale + filename);
                image.Save(Config.augmentedDataDirectoryGrayscale + filename + "\\" + count++ + ".jpg", ImageFormat.Jpeg);
            }
            else
            {
                if (!Directory.Exists(Config.augmentedDataDirectory + filename)) Directory.CreateDirectory(Config.augmentedDataDirectory + filename);
                image.Save(Config.augmentedDataDirectory + filename + "\\" + count++ + ".jpg", ImageFormat.Jpeg);
            }
        }
    }
}
