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
            Importer importer = new Importer(File.OpenText("data\\train.nmv.txt"));

            List<DecisionVariable> variables = new List<DecisionVariable>();
            int featureCount = importer.FeatureCount();
            for (int i = 0; i < featureCount; i++)
            {
                variables.Add(new DecisionVariable(string.Format("a_{0}", i), DecisionVariableKind.Continuous));
            }

            DecisionTree t = new DecisionTree(variables, 2);

            C45Learning tree = new C45Learning(t);

            var classes = importer.GetClasses();
            var matrix = importer.WithoutClass().ToArray();
            Console.WriteLine("Training DecisionTree...");
            double estimatedErrorC45 = tree.Run(matrix, classes);

            // this is just funny
            //File.WriteAllText("Best.cs", t.ToCode("BestClass"));

            Console.WriteLine("Estimated error: {0}", estimatedErrorC45);

            int correct = 0;

            for (int x = 0; x < matrix.Length; x++)
            {
                var inner = matrix[x];
                int classification = t.Compute(inner);

                if (classification == classes[x])
                {
                    correct++;
                }
            }

            Console.WriteLine("{0} correct out of {1}", correct, classes.Length);
            Console.WriteLine("Percentage: {0}", ((float)correct / (float)classes.Length) * 100);
        }
    }
}
