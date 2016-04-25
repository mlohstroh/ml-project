using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Accord.MachineLearning.Bayes;
using Accord.Statistics.Distributions.Univariate;
using Accord.Math;

namespace ml_project
{
    public class Bayes : ILearner
    {
        public void Run()
        {
            Importer i = new Importer(File.OpenText("data\\train.nmv.txt"));

            var dataTable = i.WithoutClass();

            Console.WriteLine("Row Count: {0} Colum Count: {1}", dataTable.GetLength(0), dataTable.GetLength(1));

            // Failed NBayes stuff...

            //Console.WriteLine("Normalizing...");

            //Console.WriteLine("Normalization done!");

            int colums = dataTable.GetLength(1);

            //// bagging things
            //var bagged = i.GetBaggedInputs(10000);
            //var baggedInputs = bagged.Submatrix(null, 0, bagged[0].Length - 2);
            //var baggedClasses = bagged.GetColumn(bagged[0].Length - 1).ToInt32();

            var unbaggedBayes = new NaiveBayes<NormalDistribution>(2, colums, new NormalDistribution());
            //var baggedBayes = new NaiveBayes<NormalDistribution>(2, colums, new NormalDistribution());

            Console.WriteLine("Training NBayes...");
            var classes = i.GetClasses();

            //var array = dataTable.ToArray();

            //// lets normalize it all!
            //for(int idx = 0; idx < array.Length; idx++)
            //{
            //    array[idx].Normalize(true);
            //}
            //DataGridBox.Show(dataTable, "Un-normalized");

            //DataGridBox.Show(array, "Normalized").Hold();

            var unbaggedError = unbaggedBayes.Estimate(dataTable.ToArray(), classes);

            Console.WriteLine("Estimated NBayes Error: (unbagged) {0}", unbaggedError);

            var array = dataTable.ToArray();

            int correct = 0;

            for (int x = 0; x < array.Length; x++)
            {
                var inner = array[x];
                int classification = unbaggedBayes.Compute(inner);

                if (classification == classes[x])
                {
                    correct++;
                }
            }

            Console.WriteLine("Unbagged Result");
            Console.WriteLine("{0} correct out of {1}", correct, classes.Length);
            Console.WriteLine("Percentage: {0}", ((float)correct / (float)classes.Length) * 100);
        }
    }
}
