#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;
using namespace std;


void negative(Mat& img)
{
    #pragma omp parallel for 
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            img.at<Vec3b>(i, j)[0] = 255 - img.at<Vec3b>(i, j)[0];
            img.at<Vec3b>(i, j)[1] = 255 - img.at<Vec3b>(i, j)[1];
            img.at<Vec3b>(i, j)[2] = 255 - img.at<Vec3b>(i, j)[2];
        }
    }
}

void negativeVectorization(Mat& img)
{
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    int length = width * height * channels;
    uchar* data = img.data;

    int i;
    uchar* p;
    __m128i xmm0, xmm1;

    for (i = 0, p = data; i < length; i += 16, p += 16)
    {
        xmm0 = _mm_load_si128((__m128i*)p);
        xmm1 = _mm_sub_epi8(_mm_set1_epi8(255), xmm0);
        _mm_store_si128((__m128i*)p, xmm1);
    }
}

void medianFilter(Mat& src, Mat& dst)
{
    
    dst.create(src.size(), src.type());
#pragma omp parallel for 
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            vector<Vec3b> pixels;
            for (int ii = i - 7; ii <= i + 7; ii++)
            {
                for (int jj = j - 7; jj <= j + 7; jj++)
                {
                    if (ii >= 0 && ii < src.rows && jj >= 0 && jj < src.cols)
                    {
                        pixels.push_back(src.at<Vec3b>(ii, jj));
                    }
                }
            }

            sort(pixels.begin(), pixels.end(),
                [](const Vec3b& a, const Vec3b& b)
                {
                    return a[0] < b[0];
                });
            dst.at<Vec3b>(i, j)[0] = pixels[pixels.size() / 2][0];

            sort(pixels.begin(), pixels.end(),
                [](const Vec3b& a, const Vec3b& b)
                {
                    return a[1] < b[1];
                });
            dst.at<Vec3b>(i, j)[1] = pixels[pixels.size() / 2][1];

            sort(pixels.begin(), pixels.end(),
                [](const Vec3b& a, const Vec3b& b)
                {
                    return a[2] < b[2];
                });
            dst.at<Vec3b>(i, j)[2] = pixels[pixels.size() / 2][2];
        }
    }
}
Mat  medianFilterVectorization(const Mat& image)
{
    Mat resultImage(image.rows, image.cols, image.type());

    for (int row = 0; row < image.rows; row++)
        for (int col = 0; col < image.cols; col++) {
            __m128i r = _mm_setzero_si128();
            __m128i g = _mm_setzero_si128();
            __m128i b = _mm_setzero_si128();

            for (int pxRow = row - 8; pxRow <= row + 8; pxRow++)
                for (int pxCol = col - 8; pxCol <= col + 8; pxCol++)
                    if (pxRow >= 0 && pxRow < image.rows && pxCol >= 0 && pxCol < image.cols) {
                        Vec3b pixel = image.at<Vec3b>(pxRow, pxCol);

                        __m128i pixelVec = _mm_set_epi32(pixel[3], pixel[2], pixel[1], pixel[0]);

                        r = _mm_add_epi32(r, _mm_and_si128(pixelVec, _mm_set1_epi32(0x000000FF)));
                        g = _mm_add_epi32(g, _mm_and_si128(pixelVec, _mm_set1_epi32(0x0000FF00)));
                        b = _mm_add_epi32(b, _mm_and_si128(pixelVec, _mm_set1_epi32(0x00FF0000)));
                    }

            r = _mm_srli_epi32(r, 8);
            g = _mm_srli_epi32(g, 8);
            b = _mm_srli_epi32(b, 8);

            __m128i resultVector = _mm_or_si128(_mm_or_si128(r, _mm_slli_epi32(g, 8)), _mm_slli_epi32(b, 16));

            Vec3b resultPixel;

            resultPixel[0] = (uchar)_mm_extract_epi32(resultVector, 0);
            resultPixel[1] = (uchar)_mm_extract_epi32(resultVector, 1);
            resultPixel[2] = (uchar)_mm_extract_epi32(resultVector, 2);

            resultImage.at<Vec3b>(row, col) = resultPixel;
        }
    return resultImage;
}

int main()
{
    clock_t start1 = clock();
    Mat neg = imread("C:/Users/karel/OneDrive/Рабочий стол/парадигмы/300x300.png");
    for (int i = 0; i < 1000; i++)
    {
        negative(neg);
    }
    //imshow("negative", neg);
    //waitKey();
    cout << (double(clock() - start1)) / CLOCKS_PER_SEC << endl;

   /* clock_t start2 = clock();
    Mat negV = imread("C:/Users/karel/OneDrive/Рабочий стол/парадигмы/950x950.png");
    for (int i = 0; i < 100; i++)
    {
        negativeVectorization(negV);
    }
    //imshow("negative", negV);
    //waitKey();
    cout << (double(clock() - start2)) / CLOCKS_PER_SEC << endl;*/

    clock_t start3 = clock();
    Mat median = imread("C:/Users/karel/OneDrive/Рабочий стол/парадигмы/300x300.png");
    Mat median1;
    for (int i = 0; i < 10; i++)
    {
        medianFilter(median, median1);
    }
    //imshow("medianFilter", median1);
    //waitKey();
    cout << (double(clock() - start3)) / CLOCKS_PER_SEC << endl;

    /*clock_t start4 = clock();
    Mat medianV = imread("C:/Users/karel/OneDrive/Рабочий стол/парадигмы/950x950.png");
    for (int i = 0; i < 10; i++)
    {
    Mat medianFilter=medianFilterVectorization(medianV);
    }
    //imshow("medianFilter", medianFilter);
    //waitKey();
    cout << (double(clock() - start4)) / CLOCKS_PER_SEC << endl;*/

    return 0;
}