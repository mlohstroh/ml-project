using System;
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

            Console.WriteLine("Row Count: {0} Colum Count: {1}", dataTable.Length, dataTable.GetLength(1));
            


            // Failed NBayes stuff...

            //Console.WriteLine("Normalizing...");
            
            //Console.WriteLine("Normalization done!");
    
            //var bayes = new NaiveBayes<NormalDistribution>(2, dataTable.Length, new NormalDistribution());

            //Console.WriteLine("Training NBayes...");
            //var classes = i.GetClasses();

            //var error = bayes.Estimate(dataTable.ToArray(), classes);

            //Console.WriteLine("Estimated NBayes Error: {0}", error);

            //DataGridBox.Show(bayes.Distributions);

            Console.ReadKey(true);
        }
    }
}
