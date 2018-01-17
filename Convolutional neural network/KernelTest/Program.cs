using System;
using LinearAlgebra;
using System.IO;
using System.Collections.Generic;

/*
 THIS PROGRAM USES SIMPLE LINEAR ALGEBRA LIBRARY FROM GITHUB
 https://github.com/HectorPulido/Simple_Linear_Algebra     

 AND FRUIT IMAGES DATASET FROM GITHUB
 https://github.com/Horea94/Fruit-Images-Dataset
*/

namespace KernelTest
{
    class Program
    {
        const double epsilon = 0.00001;
        const string Path = @"C:\Users\ASUS\Desktop\Convolution\KernelTest\KernelTest\Resource\Dataset\Training\";       
        static void Main(string[] args)
        {
            Random r = new Random(0);
            int batchSize = 25;

            Matrix[] Apple, Citric;

            if (Helper.LoadMatrix(out Apple, Path + "Apples.bin") 
                && Helper.LoadMatrix(out Citric, Path + "Citric.bin"))
            {
                Console.WriteLine("Data readed " + Apple.Length);
            }
            else
            {
                //-----------DATA 100x100
                string[] _Apple = Directory.GetFiles(Path + "Apple");
                _Apple = Shuffle(_Apple, r);
                string[] _Citric = Directory.GetFiles(Path + "Citric");
                _Citric = Shuffle(_Citric, r);

                Apple = new Matrix[batchSize];
                Citric = new Matrix[batchSize];

                Console.WriteLine("Data can not be readed");
                for (int i = 0; i < batchSize; i++)
                {
                    Apple[i] = ImageHelper.LoadImage(_Apple[i])[0];
                    Citric[i] = ImageHelper.LoadImage(_Citric[i])[0];
                }
                Helper.SaveMatrix(Apple, Path + "Apples.bin");
                Helper.SaveMatrix(Citric, Path + "Citric.bin");

                Console.WriteLine("Data saved");
            }


            //-----------FILTERS AND WEIGHTS
            Matrix[][] filters1 = new Matrix[6][]; // First Dimension: KERNELS, Second Dimension: Features
            for (int i = 0; i < filters1.Length; i++)
            {
                filters1[i] = new Matrix[1];
                for (int j = 0; j < filters1[i].Length; j++)
                {
                    filters1[i][j] = Matrix.Random(3, 3, r) * 2.0 - 1.0;
                }
            }
            Matrix[][] filters2 = new Matrix[6][]; // First Dimension: KERNELS, Second Dimension: Features
            for (int i = 0; i < filters2.Length; i++)
            {
                filters2[i] = new Matrix[6];
                for (int j = 0; j < filters2[i].Length; j++)
                {
                    filters2[i][j] = Matrix.Random(3, 3, r) * 2.0 - 1.0;
                }
            }
            Matrix W1 = Matrix.Random(384 + 1, 150, r) * 2.0 - 1.0;
            Matrix W2 = Matrix.Random(150 + 1, 50, r) * 2.0 - 1.0;
            Matrix W3 = Matrix.Random(50 + 1, 2, r) * 2.0 - 1.0;

            //-----------PARAMETERS

            int LayerCount = 4;
            double learningRate = 01e-15;

            Matrix[][] Z1 = new Matrix[batchSize * 2][];
            Matrix[][] Z2 = new Matrix[batchSize * 2][];
            Matrix[][] A1 = new Matrix[batchSize * 2][];
            Matrix[][] A2 = new Matrix[batchSize * 2][];
            Matrix X = new Matrix(384, batchSize * 2);
            Matrix Y = new Matrix(batchSize * 2, 2);

            Matrix[] error = new Matrix[LayerCount];
            Matrix[] delta = new Matrix[LayerCount];

            for (int i = 0; i < batchSize; i++)
            {
                Y[i, 0] = 1;
                Y[i, 1] = 0;
            }
            for (int i = batchSize; i < batchSize * 2; i++)
            {
                Y[i, 0] = 0;
                Y[i, 1] = 1;
            }

            for (int epoch = 0; epoch < 30001; epoch++)
            {
                //-----------FORWARDPROPAGATION
                for (int i = 0; i < batchSize; i++)
                {
                    ConvolutionLayer(out Z1[i], out A1[i], new Matrix[] { Apple[i] }, filters1);
                    A1[i] = Pooling(A1[i], 3);

                    ConvolutionLayer(out Z2[i], out A2[i], A1[i], filters2);
                    A2[i] = Pooling(A2[i], 4);

                    for (int j = 0; j < A2[i].Length; j++)
                    {
                        for (int k = 0; k < A2[i][j].flat.x; k++)
                        {
                            X[j * A2.Length + k, i] = A2[i][j].flat[k, 0]; // 384 per image
                        }
                    }
                }
                for (int i = batchSize; i < batchSize * 2; i++)
                {
                    ConvolutionLayer(out Z1[i], out A1[i], new Matrix[] { Citric[i - batchSize] }, filters1);
                    A1[i] = Pooling(A1[i], 3);

                    ConvolutionLayer(out Z2[i], out A2[i], A1[i], filters2);
                    A2[i] = Pooling(A2[i], 4);

                    for (int j = 0; j < A2[i].Length; j++)
                    {
                        for (int k = 0; k < A2[i][j].flat.x; k++)
                        {
                            X[j * A2.Length + k, i] = A2[i][j].flat[k, 0]; // 384 per image
                        }
                    }
                }

                Matrix[] Z, A;
                Matrix[] W = new Matrix[] { W1, W2, W3 };

                FullyConectedLayer(out Z, out A, W, X.T);

                Matrix output = Z[Z.Length - 1].Slice(0, 1, Z[Z.Length - 1].x, Z[Z.Length - 1].y);
                
                //BACKPROPAGATION
                error[error.Length - 1] = output - Y;
                delta[delta.Length - 1] = error[error.Length - 1] * Relu(output, true);

                for (int i = LayerCount - 2; i >= 0; i--)
                {
                    error[i] = delta[i + 1] * W[i].T;
                    delta[i] = error[i] * Relu(Z[i], true);
                    delta[i] = delta[i].Slice(0, 1, delta[i].x, delta[i].y);
                }



                //Gradient Descend
                W1 += (A[0].T * delta[0 + 1]) * learningRate;
                W2 += (A[1].T * delta[1 + 1]) * learningRate;
                W3 += (A[2].T * delta[2 + 1]) * learningRate;


                Console.WriteLine(error[error.Length - 1].abs.average * batchSize * 2);
                Console.WriteLine("Epoch " + epoch);

            }     

            Console.WriteLine("Press any key...");
            Console.ReadKey();

        }
        static Matrix[] Pooling(Matrix[] image, int filterSize)
        {
            Matrix[] _output = new Matrix[image.Length];
            for (int i = 0; i < image.Length; i++)
            {
                _output[i] = new double[image[0].x / filterSize, image[0].y / filterSize];
            }
            for (int i = 0; i < image[0].x - filterSize; i += filterSize)
            {
                for (int j = 0; j < image[0].y - filterSize; j += filterSize)
                {
                    for (int k = 0; k < image.Length; k++)
                    {
                        double pool = image[k].Slice(i, j, i + filterSize, j + filterSize).max;
                        _output[k].SetValue(i / filterSize, j / filterSize, pool);
                    }
                }
            }            
            return _output;
        }
        static Matrix[] Convolution(Matrix[] image, Matrix[][] kernel)
        {
            Matrix[] output = new Matrix[kernel.Length];
            for (int i = 0; i < kernel.Length; i++)
            {
                output[i] = Convolution(image, kernel[i]);
            }
            return output;
        }
        static Matrix Convolution(Matrix[] image, Matrix[] kernel)
        {
            Matrix _output = new Matrix(image[0].x, image[0].y);
            for (int i = kernel[0].x / 2; i < image[0].x - kernel[0].x / 2; i++) // MUST NO TOUCH THE BORDERS
            {
                for (int j = kernel[0].x / 2; j < image[0].x - kernel[0].x / 2; j++)
                {
                    double temp = 0;
                    for (int k = 0; k < image.Length; k++)
                    {
                        for (int l = 0; l < kernel.Length; l++)
                        {
                            temp += (image[k].Slice(
                                                i - kernel[l].x / 2,
                                                j - kernel[l].x / 2,
                                                i + kernel[l].x / 2 + 1,
                                                j + kernel[l].x / 2 + 1) *
                                                kernel[l]).Sumatory()[0,0];
                        }
                    }                   
                    _output[i,j] = temp;
                }
            }             
            return _output;
        }

        static void ConvolutionLayer(out Matrix[] Z, out Matrix[] A, 
                                    Matrix[] image, Matrix[][] filter)
        {
            Z = Convolution(image , filter);
            A = Relu(Z);
        }

        static void FullyConectedLayer(out Matrix[] Z, out Matrix[] A, 
                                        Matrix[] W, Matrix InputValue)
        {
            int LayerCount = W.Length + 1;
            int ExampleCount = InputValue.x;
            Z = new Matrix[LayerCount];
            A = new Matrix[LayerCount];

            Z[0] = InputValue.AddColumn(Matrix.Ones(InputValue.x, 1));
            A[0] = Z[0];

            for (int i = 1; i < LayerCount; i++)
            {
                Z[i] = (A[i - 1] * W[i - 1]).AddColumn(Matrix.Ones(A[i - 1].x, 1));
                A[i] = Relu(Z[i]);
            }
        }
        static Matrix Relu(Matrix m, bool derivated = false)
        {
            double[,] output = new double[m.x, m.y];
            Matrix.MatrixLoop((i, j) => {
                if (derivated)
                    output[i, j] = m[i,j] > 0 ? 1 : epsilon;
                else
                    output[i, j] = m[i, j] > 0 ? m[i, j] : 0;
            }, m.x, m.y);
            return output;
        }
        static Matrix[] Relu(Matrix[] m, bool derivated = false)
        {
            Matrix[] output = new Matrix[m.Length];
            for (int i = 0; i < m.Length; i++)
            {
                output[i] = Relu(m[i], derivated);
            }            
            return output;
        }
        static Matrix HardRandom(int x, int y, double p, Random r)
        {
            double[,] HR = new double[x, y];
            Matrix.MatrixLoop((i, j) => {
                HR[i, j] = (r.NextDouble() < p) ? 1 : 0;
            }, x, y);
            return HR;
        }
        static string[] Shuffle(string[] list, Random r)
        {
            int n = list.Length;
            while (n > 1)
            {
                n--;
                int k = r.Next(n + 1);
                string value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
            return list;
        }

    }
}
