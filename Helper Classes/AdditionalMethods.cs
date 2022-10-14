using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using Microsoft.ML;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Face_Detection_and_Recognition_Server_V2.Helper_Classes
{
    internal class AdditionalMethods
    {
        public static string nameOfTrainedModel(int modelChoice, int augmentationChoice)
        {
            
            if (modelChoice == 0 && (augmentationChoice == 0 || augmentationChoice == 2))
            {
                return "ResnetV250Colored";
            }
            else if (modelChoice == 1 && (augmentationChoice == 0 || augmentationChoice == 2))
            {
                return "ResnetV2101Colored";
            }
            else if (modelChoice == 2 && (augmentationChoice == 0 || augmentationChoice == 2))
            {
                return "InceptionV3Colored";
            }
            else if (modelChoice == 3 && (augmentationChoice == 0 || augmentationChoice == 2))
            {
                return "MobilenetV2Colored";
            }
            else if (modelChoice == 0 && (augmentationChoice == 1 || augmentationChoice == 3))
            {
                return "ResnetV250Grayscale";
            }
            else if (modelChoice == 1 && (augmentationChoice == 1 || augmentationChoice == 3))
            {
                return "ResnetV2101Grayscale";
            }
            else if (modelChoice == 2 && (augmentationChoice == 1 || augmentationChoice == 3))
            {
                return "InceptionV3Grayscale";
            }
            else if (modelChoice == 3 && (augmentationChoice == 1 || augmentationChoice == 3))
            {
                return "MobilenetV2Grayscale";
            }

            return "Invalid Input";
        }

        public static void showPredictionsAndSaveModel(string modelName, MLContext mlContext, IDataView testSet, ITransformer trainedModel, DataViewSchema trainSetSchema)
        {
            ConsoleWriteHeader("*** Showing single prediction ***");
            ClassifySingleImage(mlContext, testSet, trainedModel);

            ConsoleWriteHeader("*** Showing multiple predictions ***");
            ClassifyImages(mlContext, testSet, trainedModel, modelName);

            var watch = System.Diagnostics.Stopwatch.StartNew();
            ConsoleWriteHeader($"*** Save {modelName} model to local file ***");

            mlContext.Model.Save(trainedModel, trainSetSchema, Config.serverDotNetFaceRecognitionModelsDirectory + $"\\{modelName}.zip");
            mlContext.Model.Save(trainedModel, trainSetSchema, Config.clientFaceRecongitionModelsDirectory + modelName + ".zip");

            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine($"Saving {modelName} took: " + (elapsedMs / 1000).ToString() + " seconds");
        }
        public static EstimatorChain<KeyToValueMappingTransformer> trainingPipeline(MLContext mlContext, IDataView validationSet, int modelChoice)
        {
            return mlContext.MulticlassClassification.Trainers.ImageClassification(createClassifierOptions(modelChoice, validationSet))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }
        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new String('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static ImageClassificationTrainer.Options createClassifierOptions(int modelChoice, IDataView validationSet)
        {
            ImageClassificationTrainer.Architecture model = ImageClassificationTrainer.Architecture.ResnetV250;
            if (modelChoice == 1) model = ImageClassificationTrainer.Architecture.ResnetV2101;
            else if (modelChoice == 2) model = ImageClassificationTrainer.Architecture.InceptionV3;
            else if (modelChoice == 3) model = ImageClassificationTrainer.Architecture.MobilenetV2;

            return new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                //ValidationSetFraction = 0.1f,
                Arch = model,
                EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping(0.01f, 10, ImageClassificationTrainer.EarlyStoppingMetric.Loss, false),
                Epoch = 1000000,
                BatchSize = 8, //8, 16, 32, 64
                LearningRate = 0.01f, // 0.01, 0.001, 0.0001
                LearningRateScheduler = new Microsoft.ML.Trainers.ExponentialLRDecay(learningRate: 0.01f),
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                //WorkspacePath = Config.tensorflowFaceRecognitionModelsDirectory
            };
        }

        public static void ClassifyUploadededImage(MLContext mlContext, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            Mat img = new Mat("path here");

            ModelInput sampleData = new ModelInput()
            {
                Image = img.ToBytes(),
            };

            ModelOutput prediction = predictionEngine.Predict(sampleData);

            Console.WriteLine("Classifying uploaded image");
            OutputPrediction(prediction);
        }

        public static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();

            ModelOutput prediction = predictionEngine.Predict(image);

            Console.WriteLine("Classifying single image");
            OutputPrediction(prediction);
        }

        public static void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel, string modelName)
        {
            IDataView predictionData = trainedModel.Transform(data);

            IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);

            Console.WriteLine("Classifying multiple images");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }

            ConsoleWriteHeader("Classification metrics");
            MulticlassClassificationMetrics trainedModelMetrics = mlContext.MulticlassClassification.Evaluate(predictionData, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {modelName} an image classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {trainedModelMetrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {trainedModelMetrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLossReduction = {trainedModelMetrics.LogLossReduction:0.####}, a value between  -inf and 1.00, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {trainedModelMetrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {trainedModelMetrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {trainedModelMetrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }

        private static void OutputPrediction(ModelOutput prediction)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel} | Predicted Value Score: {prediction.Score.Max() * 100}%");
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameasLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameasLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };

            }
        }
    }
}
