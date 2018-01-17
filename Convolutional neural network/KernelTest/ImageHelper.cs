using System;
using System.Drawing;
using System.Drawing.Imaging;
namespace LinearAlgebra
{
    public static class ImageHelper
    {

        static Color c;
        static Bitmap img;

        public static Bitmap ResizeImage(Bitmap _img, double SizeX, double SizeY)
        {
            var brush = new SolidBrush(Color.Black);
            float scale = Math.Min((float)SizeX / _img.Width, (float)SizeY / _img.Height);

            img = new Bitmap((int)SizeX, (int)SizeY);

            var graph = Graphics.FromImage(img);
            int scaleWidth = (int)(_img.Width * scale);
            int scaleHeight = (int)(_img.Height * scale);
            graph.FillRectangle(brush, new RectangleF(0, 0, (float)SizeX, (float)SizeY));
            graph.DrawImage(_img, new Rectangle((int)((SizeX - scaleWidth) / 2.0), (int)((SizeY - scaleHeight) / 2.0), (int)SizeX, (int)SizeY));

            return img;
        }
        public static Matrix[] ImageToMatrix(Bitmap img)
        {
            int h = img.Height;
            int w = img.Width;
            double[,] r = new double[h, w];
            double[,] g = new double[h, w];
            double[,] b = new double[h, w];
            double[,] a = new double[h, w];
            double[,] bw = new double[h, w];
            Matrix.MatrixLoop((i, j) => {
                c = img.GetPixel(i, j);
                r[i, j] = c.R;
                g[i, j] = c.G;
                b[i, j] = c.B;
                a[i, j] = c.A;
                bw[i, j] = c.GetBrightness() * 255;
            }, h, w);
            return new Matrix[] { bw, r, g, b };
        }
        public static Matrix[] LoadImage(string directory, double SizeX, double SizeY)
        {
            img = ResizeImage((Bitmap)Image.FromFile(directory), SizeX, SizeY);
            return ImageToMatrix(img);
        }
        public static Matrix[] LoadImage(string directory)
        {
            img = (Bitmap)Image.FromFile(directory);
            return ImageToMatrix(img);
        }
        public static void SaveImage(double[,] bw, string directory)
        {
            img = new Bitmap(bw.GetLength(0), bw.GetLength(1));
            Matrix.MatrixLoop((i, j) => {
                c = Color.FromArgb(
                    (int)clamp(bw[i, j], 255, 0), 
                    (int)clamp(bw[i, j], 255, 0), 
                    (int)clamp(bw[i, j], 255, 0));
                img.SetPixel(i, j, c);
            }, bw.GetLength(0), bw.GetLength(1));
            img.Save(directory, ImageFormat.Bmp);
        }
        public static void SaveImage(double[,] r, double[,] g, double[,] b, string directory)
        {
            img = new Bitmap(r.GetLength(0), r.GetLength(1));
            Matrix.MatrixLoop((i, j) => {
                c = Color.FromArgb(
                    (int)clamp(r[i, j], 255, 0), 
                    (int)clamp(g[i, j], 255, 0), 
                    (int)clamp(b[i, j], 255, 0));
                img.SetPixel(i, j, c);
            }, r.GetLength(0), r.GetLength(1));
            img.Save(directory, ImageFormat.Bmp);
        }
        public static void SaveImage(double[,] a, double[,] r, 
            double[,] g, double[,] b, string directory)
        {
            img = new Bitmap(r.GetLength(0), r.GetLength(1));
            Matrix.MatrixLoop((i, j) => {
                c = Color.FromArgb(
                    (int)clamp(a[i, j], 255, 0),
                    (int)clamp(r[i, j], 255, 0),
                    (int)clamp(g[i, j], 255, 0),
                    (int)clamp(b[i, j], 255, 0));
                img.SetPixel(i, j, c);
            }, r.GetLength(0), r.GetLength(1));
            img.Save(directory, ImageFormat.Bmp);
        }
        static double clamp(double d, double max, double min)
        {
            return d > max ? max : d < min ? min : d;
        }

    }
}
