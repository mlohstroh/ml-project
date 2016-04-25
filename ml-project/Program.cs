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

            ILearner bayes = new Bayes();

            bayes.Run();

            ILearner tree = new DTree();

            tree.Run();

            //DataGridBox.Show(bayes.Distributions).Hold();
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey(true);
        }
    }
}
