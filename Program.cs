using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;
using Face_Detection_and_Recognition_Server_V2.Helper_Classes;

namespace Face_Detection_and_Recognition_Server_V2
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int augmentationChoice;
            Console.WriteLine("Select Image Augmentation Process: ");
            Console.WriteLine("0 = Colored Image Augmentation ");
            Console.WriteLine("1 = Grayscale Image Augmentation  ");
            Console.WriteLine("2 = Colored Image Augmentation Done ");
            Console.WriteLine("3 = Grayscale Image Augmentation Done ");
            do
            {
                Console.Write("Enter choice: ");
                augmentationChoice = int.Parse(Console.ReadKey().KeyChar.ToString());
                Console.WriteLine();
            } while (augmentationChoice != 0 && augmentationChoice != 1 && augmentationChoice != 2 && augmentationChoice != 3);


            if (augmentationChoice == 0 || augmentationChoice == 1) ImageAugmentation.runImageAugmentation(augmentationChoice);

            MLContext mlContext = new MLContext();

            var watch = System.Diagnostics.Stopwatch.StartNew();
            AdditionalMethods.ConsoleWriteHeader("*** Load Dataset ***");

            IEnumerable<ImageData> images = null;
            if (augmentationChoice == 0 || augmentationChoice == 2) images = AdditionalMethods.LoadImagesFromDirectory(folder: Config.augmentedDataDirectory, useFolderNameasLabel: true);
            else if (augmentationChoice == 1 || augmentationChoice == 3) images = AdditionalMethods.LoadImagesFromDirectory(folder: Config.augmentedDataDirectoryGrayscale, useFolderNameasLabel: true);

            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Loading Dataset took: " + (elapsedMs / 1000).ToString() + " seconds");

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                                                                        inputColumnName: "Label",
                                                                        outputColumnName: "LabelAsKey")
                                        .Append(mlContext.Transforms.LoadRawImageBytes(
                                                                        outputColumnName: "Image",
                                                                        imageFolder: (augmentationChoice == 0 || augmentationChoice == 2)? Config.augmentedDataDirectory : Config.augmentedDataDirectoryGrayscale,
                                                                        inputColumnName: "ImagePath"));

            watch = System.Diagnostics.Stopwatch.StartNew();
            AdditionalMethods.ConsoleWriteHeader("*** Preproccesing Data Before Training ***");

            IDataView preProcessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);

            watch.Stop();
            elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Preproccesing Data took: " + (elapsedMs / 1000).ToString() + " seconds");

            watch = System.Diagnostics.Stopwatch.StartNew();
            AdditionalMethods.ConsoleWriteHeader("*** Split Dataset to Train and Test Data ***");

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;

            watch.Stop();
            elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Splitting Dataset took: " + (elapsedMs / 1000).ToString() + " seconds");
            
            int modelChoice;
            Console.WriteLine("Select Model: ");
            Console.WriteLine("0 = ResnetV250 ");
            Console.WriteLine("1 = ResnetV2101 ");
            Console.WriteLine("2 = InceptionV3 ");
            Console.WriteLine("3 = MobilenetV2 ");
            do
            {
                Console.Write("Enter choice: ");
                modelChoice = int.Parse(Console.ReadKey().KeyChar.ToString());
                Console.WriteLine();
            } while (modelChoice != 0 && modelChoice != 1 && modelChoice != 2 && modelChoice != 3);

            watch = System.Diagnostics.Stopwatch.StartNew();
            AdditionalMethods.ConsoleWriteHeader("*** Training " + AdditionalMethods.nameOfTrainedModel(modelChoice, augmentationChoice) + " model ***");

            ITransformer trainedModel = AdditionalMethods.trainingPipeline(mlContext, validationSet, modelChoice).Fit(trainSet);

            watch.Stop();
            elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Training with " + AdditionalMethods.nameOfTrainedModel(modelChoice, augmentationChoice) + " model took: " + (elapsedMs / 1000).ToString() + " seconds");

            AdditionalMethods.showPredictionsAndSaveModel(AdditionalMethods.nameOfTrainedModel(modelChoice, augmentationChoice), mlContext, testSet, trainedModel, trainSet.Schema);

            Console.WriteLine("Done Training");
            Console.ReadKey();
        }
    }
}
