﻿using System;
using System.IO;
using Accord.Math;
using Accord.Statistics.Filters;
using Accord.IO;
using Accord.Collections;
using Accord.Controls;
using Accord.MachineLearning.Bayes;
using Accord.Statistics.Distributions.Univariate;
using System.Threading.Tasks;

namespace ml_project
{
    class Program
    {
        static void Main(string[] args)
        {
            Importer i = new Importer(File.OpenText("data\\train.nmv.txt"));

            var dataTable = i.WithoutClass();

            Console.WriteLine("Row Count: {0} Colum Count: {1}", dataTable.GetLength(0), dataTable.GetLength(1));

            // Failed NBayes stuff...

            //Console.WriteLine("Normalizing...");

            //Console.WriteLine("Normalization done!");

            int colums = dataTable.GetLength(1);

            var bayes = new NaiveBayes<NormalDistribution>(2, colums, new NormalDistribution());

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

            var error = bayes.Estimate(dataTable.ToArray(), classes);

            Console.WriteLine("Estimated NBayes Error: {0}", error);

            var array = dataTable.ToArray();

            int correct = 0;

            for(int x = 0; x < array.Length; x++)
            {
                var inner = array[x];
                int classification = bayes.Compute(inner);

                if(classification == classes[x])
                {
                    correct++;
                }
            }

            Console.WriteLine("{0} correct out of {1}", correct, classes.Length);
            Console.WriteLine("Percentage: {0}", ((float)correct / (float)classes.Length) * 100);


            //DataGridBox.Show(bayes.Distributions).Hold();
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey(true);
        }
    }
}
