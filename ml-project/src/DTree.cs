using Accord.MachineLearning.DecisionTrees;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.Math;

namespace ml_project
{
    public class DTree : ILearner
    {
        public void Run()
        {
            Importer testData = new Importer(File.OpenText("data\\final-nmv-noclass.txt"));

            Importer importer = new Importer(File.OpenText("data\\train.nmv.txt"));

            List<DecisionVariable> variables = new List<DecisionVariable>();
            int featureCount = importer.FeatureCount();
            for (int i = 0; i < featureCount; i++)
            {
                variables.Add(new DecisionVariable(string.Format("a_{0}", i), DecisionVariableKind.Continuous));
            }

            DecisionTree @base = new DecisionTree(variables, 2);
            DecisionTree trimmed = new DecisionTree(variables, 2);

            C45Learning tree = new C45Learning(@base)
            {
                Join = 1,
                MaxHeight = 75,
                // maybe set a max height?
            };

            C45Learning trimmedLearner = new C45Learning(trimmed)
            {
                Join = 1,
                MaxHeight = 80,
            };

            var trainMatrix = importer.GetBaggedInputs(5000);
            var trainClasses = trainMatrix.GetColumn(trainMatrix[0].Length - 1).ToInt32();
            var trainInputs = trainMatrix.Submatrix(null, 0, trainMatrix[0].Length - 2);

            var classes = importer.GetClasses();
            var matrix = importer.WithoutClass().ToArray();

            Console.WriteLine("Training DecisionTree...");
            double estimatedErrorC45 = tree.Run(matrix, classes);

            //Console.WriteLine("Training Trimmed DecisionTree...");
            //double estimatedErrorTrimmed = trimmedLearner.Run(trainInputs, trainClasses);

            // this is just funny
            //File.WriteAllText("Best.cs", t.ToCode("BestClass"));

            Console.WriteLine("Estimated error: {0}", estimatedErrorC45);
            //Console.WriteLine("Height of finished tree: {0}", t.GetHeight
            //Console.WriteLine("Estimated error (trimmed): {0}", estimatedErrorTrimmed);

            int correctAll = 0;
            int correctTrimmed = 0;

            for (int x = 0; x < matrix.Length; x++)
            {
                var inner = matrix[x];
                int classification = @base.Compute(inner);

                if (classification == classes[x])
                {
                    correctAll++;
                }

                //classification = trimmed.Compute(inner);
                //if(classification == classes[x])
                //{
                  //  correctTrimmed++;
                //}
            }

            Console.WriteLine("{0} correct out of {1} (all)", correctAll, classes.Length);
            Console.WriteLine("Percentage: {0} (all)", ((float)correctAll / (float)classes.Length) * 100);

            //Console.WriteLine("{0} correct out of {1} (trimmed)", correctTrimmed, classes.Length);
            //Console.WriteLine("Percentage: {0} (trimmed)", ((float)correctTrimmed / (float)classes.Length) * 100);


            //// I need with the class because this test data doesn't include the class
            var testMatrix = testData.WithClass().ToArray();

            List<int> basePredictions = new List<int>();

            //int differences = 0;

            for(int idx = 0; idx < testMatrix.Length; idx++)
            {
                var inner = testMatrix[idx];
                int classification = @base.Compute(inner);
                //int trimmedClass = trimmed.Compute(inner);
                basePredictions.Add(classification);
            }

            //Console.WriteLine("Differences: {0} out of {1} inputs", differences, testMatrix.Length);

            if(!Directory.Exists("output"))
                Directory.CreateDirectory("output");

            using (StreamWriter writer = new StreamWriter("output\\prediction.txt"))
            {
                foreach (int i in basePredictions)
                {
                    writer.WriteLine(i);
                }
            }
        }
    }
}
