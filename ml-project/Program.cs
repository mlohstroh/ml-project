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

            //ILearner bayes = new Bayes();

            //bayes.Run();

            ILearner tree = new DTree();

            tree.Run();

            //ILearner svm = new SVM();
            //svm.Run();

            //DataGridBox.Show(bayes.Distributions).Hold();
            Console.WriteLine("Type quit to continue...");
            string exit = null;
            while ((exit = Console.ReadLine()) != "quit") 
            { }
        }
    }
}
