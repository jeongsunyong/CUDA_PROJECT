
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int keycnt = 0;
///// A S S I G N M E N T ///////////////////////////////////////////
//
//#1. INPUT
//#   움직임이 존재하는 2장의 연속 프레임영상을 입력으로 사용
//
//#2. CORNER DETECTION
//#   CORNER DETECTION을 이용해 2장의 영상 중 앞쪽 영상의 Key Point를 도출
//#Harris Featrue
//#Hog Discriptor
//#SIFT
//
//#Harris Corner Detector
//
//#3. MOTION VECTOR
//#   OPTICAL FLOW를 이용하여 앞쪽 영상의 KEYPOINT픽셀의 MOTION VECTOR(x, y)를 도출.
//
//#   - Keypoint 도출 시 조정 가능 파라미터 값 재량껏
//#   - 파라미터 조정에 따른 성능변화 기입
//#   - RGB픽셀당 한개의 detection결과가 도출되게 함
//
//#   - RGB픽셀당 한개의 motion이 도출됨
//#   -모션 도출 결과를 영상에 효과적으로 표현
//#   -조정 가능한 파라미터들의 정의는 작성자 본인이 수행
//#   - 파라미터 조정에 따른 성능변화 기입
//
//
//
//
//#EVALUATION
//#code 개발 환경 기술(5)
//#알고리즘 flow chart로 표현(5)
//#이론적 배경과 동작 설명(20)
//#코드 설명(20)
////////////////////////////////////////////////////////////////////
Mat grayscale(Mat frame, int w, int h); //basic
float Gaussian(int u, int v, float sigma);//basic
Mat GaussianFilter(Mat frame, int w, int h, float sigma);//basic
Mat Resize(Mat frame, float ratio, int w, int h);//basic
void MakeGaussians(Mat** Octaves, Mat frame, int level, int size, int w, int h);//1.multi-sclae-extrema detectiojn
void MakeDoG(Mat** DoG, Mat**Octaves, int level, int size);// 1.multi - sclae - extrema detectiojn

Mat finddelta(Mat** DoG, int x, int y, int lv, int s, int w, int h);//2.keypoint localization
void FindLocalExtrema(Mat** DoG, int level, int size, int w, int h, int** idx);//1.multi - sclae - extrema detectiojn
Mat orientAssignment(Mat Mask_f, int w, int h, int** idx, Mat* m, Mat* theta, Mat** octaves);
void motionExtraction(Mat frame1, Mat frame2, int cnt, int** idx, int** idx_aft, int h, int w);


int main()
{
	//1.input
	//Mat a = Mat({ 0,1,2,3 });
	//cout << a.at<int>(1,0)<< endl;

	int w = 640;
	int h = 360;
	float sigma = 1;
	Mat frame1 = imread("cat_0.jpg", IMREAD_COLOR);
	Mat frame2 = imread("cat_1.jpg", IMREAD_COLOR);
	Mat frame3 = imread("cat_2.jpg", IMREAD_COLOR);
	Mat frame4 = imread("cat_3.jpg", IMREAD_COLOR);
	Mat frame5 = imread("cat_4.jpg", IMREAD_COLOR);
	if (frame1.empty() || frame2.empty())
	{
		cout << "NOT FIND" << endl;
		return -1;
	}

	//2.Multi-Scale-Extrema - Detection

	//Mat frame_g = grayscale(frame1, w, h);
	Mat frame_g;
	Mat frame_g2;
	Mat frame_g3;
	Mat frame_g4;
	Mat frame_g5;
	cv::cvtColor(frame1, frame_g, COLOR_RGB2GRAY);
	cv::cvtColor(frame2, frame_g2, COLOR_RGB2GRAY);
	cv::cvtColor(frame3, frame_g3, COLOR_RGB2GRAY);
	cv::cvtColor(frame4, frame_g4, COLOR_RGB2GRAY);
	cv::cvtColor(frame5, frame_g5, COLOR_RGB2GRAY);


	Mat** octaves = new Mat*[4];
	for (int i = 0; i < 4; i++)
	{
		octaves[i] = new Mat[5];
	}
	Mat** DoG = new Mat*[4];
	for (int i = 0; i < 4; i++)
	{
		DoG[i] = new Mat[4];
	}
	MakeGaussians(octaves, frame_g, 4, 5, w, h);//level:4 size:5
	MakeDoG(DoG, octaves, 4, 5);


	//3.Keypint Localization ( scale-space-extreama)
	int** idx = new int*[h*w]; //keypoint idx나타내기위함, [0][0]:cnt
	for (int i = 0; i < h*w; i++)
	{
		idx[i] = new int[3];
	}
	FindLocalExtrema(DoG, 4, 4, w, h, idx); //size : 4임, DoG의 Size이므로 //

	cout << idx[0][0] << endl;
	int fcnt = idx[0][0];//feature : 1~fcnt까지 존재


	//OPtical Flow
	Mat flows = Mat::zeros(h, w, CV_8UC3);

	int** idx_aft = new int*[fcnt + 1];
	for (int i = 0; i < fcnt + 1; i++)
	{
		idx_aft[i] = new int[2];
	}
	motionExtraction(frame_g, frame_g2, fcnt, idx, idx_aft, h, w);
	for (int i = 1; i <= fcnt; i++)
	{
		int x_pre = idx[i][1];
		int y_pre = idx[i][0];
		circle(frame1, Point(x_pre, y_pre), int(1), Scalar(255, 0, 0));
		circle(frame2, Point(x_pre, y_pre), int(1), Scalar(255, 0, 0));
		int x = min(max(0, int(idx[i][1]) + int(idx_aft[i][1])), w - 1);
		int y = min(max(int(idx[i][0]) + int(idx_aft[i][0]), 0), h - 1);
		circle(frame1, Point(x, y), int(1), Scalar(200, 50, 0));
		circle(frame2, Point(x, y), int(1), Scalar(200, 50, 0));
		line(frame1, Point(x_pre, y_pre), Point(x, y), (200, 50, 0), 1);
		line(frame2, Point(x_pre, y_pre), Point(x, y), (200, 50, 0), 1);

		line(flows, Point(x_pre, y_pre), Point(x, y), (200, 50, 0), 1);
		circle(flows, Point(x_pre, y_pre), int(1), Scalar(255, 0, 0));
		circle(flows, Point(x, y), int(1), Scalar(200, 50, 0));


		idx_aft[i][1] = x;
		idx_aft[i][0] = y;
	}


	//2
	int** idx_aft2 = new int*[fcnt + 1];
	for (int i = 0; i < fcnt + 1; i++)
	{
		idx_aft2[i] = new int[2];
	}
	motionExtraction(frame_g2, frame_g3, fcnt, idx_aft, idx_aft2, h, w);
	for (int i = 1; i <= fcnt; i++)
	{

		int x_pre = idx_aft[i][1];
		int y_pre = idx_aft[i][0];
		cout << i << " x" << x_pre << endl;
		circle(frame3, Point(x_pre, y_pre), int(1), Scalar(200, 50, 0));
		int x = min(max(0, int(idx_aft[i][1]) + int(idx_aft2[i][1])), w - 1);
		int y = min(max(int(idx_aft[i][0]) + int(idx_aft2[i][0]), 0), h - 1);
		/*cout << x << "," << y << endl;*/
		circle(frame2, Point(x, y), int(1), Scalar(150, 100, 0));
		circle(frame3, Point(x, y), int(1), Scalar(150, 100, 0));
		line(frame2, Point(x_pre, y_pre), Point(x, y), (150, 100, 0), 1);
		line(frame3, Point(x_pre, y_pre), Point(x, y), (150, 100, 0), 1);

		line(flows, Point(x_pre, y_pre), Point(x, y), (150, 100, 0), 1);
		circle(flows, Point(x, y), int(1), Scalar(150, 100, 0));

		idx_aft2[i][1] = x;
		idx_aft2[i][0] = y;
	}
	//3
	int** idx_aft3 = new int*[fcnt + 1];
	for (int i = 0; i < fcnt + 1; i++)
	{
		idx_aft3[i] = new int[2];
	}
	motionExtraction(frame_g3, frame_g4, fcnt, idx_aft2, idx_aft3, h, w);
	for (int i = 1; i <= fcnt; i++)
	{

		int x_pre = idx_aft2[i][1];
		int y_pre = idx_aft2[i][0];
		cout << i << " x" << x_pre << endl;
		circle(frame4, Point(x_pre, y_pre), int(1), Scalar(150, 100, 0));
		int x = min(max(0, int(idx_aft2[i][1]) + int(idx_aft3[i][1])), w - 1);
		int y = min(max(int(idx_aft2[i][0]) + int(idx_aft3[i][0]), 0), h - 1);
		/*cout << x << "," << y << endl;*/
		circle(frame3, Point(x, y), int(1), Scalar(100, 150, 255));
		circle(frame4, Point(x, y), int(1), Scalar(100, 150, 255));
		line(frame3, Point(x_pre, y_pre), Point(x, y), (100, 150, 0), 1);
		line(frame4, Point(x_pre, y_pre), Point(x, y), (100, 150, 0), 1);

		line(flows, Point(x_pre, y_pre), Point(x, y), (100, 150, 0), 1);
		circle(flows, Point(x, y), int(1), Scalar(100, 150, 0));

		idx_aft3[i][1] = x;
		idx_aft3[i][0] = y;
	}

	//4
	int** idx_aft4 = new int*[fcnt + 1];
	for (int i = 0; i < fcnt + 1; i++)
	{
		idx_aft4[i] = new int[2];
	}
	motionExtraction(frame_g4, frame_g5, fcnt, idx_aft3, idx_aft4, h, w);
	for (int i = 1; i <= fcnt; i++)
	{

		int x_pre = idx_aft3[i][1];
		int y_pre = idx_aft3[i][0];
		cout << i << " x" << x_pre << endl;
		circle(frame5, Point(x_pre, y_pre), int(1), Scalar(100, 150, 0));
		int x = min(max(0, int(idx_aft3[i][1]) + int(idx_aft4[i][1])), w - 1);
		int y = min(max(int(idx_aft3[i][0]) + int(idx_aft4[i][0]), 0), h - 1);
		/*cout << x << "," << y << endl;*/
		circle(frame4, Point(x, y), int(1), Scalar(50, 255, 255));
		circle(frame5, Point(x, y), int(1), Scalar(50, 255, 255));
		line(frame4, Point(x_pre, y_pre), Point(x, y), (50, 255, 0), 1);
		line(frame5, Point(x_pre, y_pre), Point(x, y), (50, 255, 0), 1);


		line(flows, Point(x_pre, y_pre), Point(x, y), (50, 255, 0), 1);
		circle(flows, Point(x, y), int(1), Scalar(50, 255, 0));

		idx_aft4[i][1] = x;
		idx_aft4[i][0] = y;
	}


	imshow("1", frame1);
	imshow("2", frame2);
	imshow("3", frame3);
	imshow("4", frame4);
	imshow("5", frame5);
	imshow("flows", flows);
	waitKey(0);


	//}

	waitKey(0);
	return 0;
}






Mat grayscale(Mat frame, int w, int h)
{
	//float coef_r = 0.299;
	//float coef_g = 0.587;
	//float coef_b = 0.114;
	Mat result(h, w, CV_8UC1);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{

			int r = frame.at<Vec3b>(i, j)[0];
			int g = frame.at<Vec3b>(i, j)[1];
			int b = frame.at<Vec3b>(i, j)[2];
			result.at<uchar>(i, j) = int((r*0.299 + g * 0.587 + b * 0.114));
			//result.at<uchar>(i, j) = int((r + g + b) / 3);
		}
	}

	return result;
}
float Gaussian(int u, int v, float sigma)
{
	float pi = 3.14;
	float result = (
		(1 / (2 * pi * pow(sigma, 2))) *
		exp(
		(-1 * (pow(u, 2) + pow(v, 2))
			) / (2 * pow(sigma, 2))
		)
		);

	return result;
}
Mat GaussianFilter(Mat frame, int w, int h, float sigma)
{
	Mat result = frame.clone();
	if (sigma == 0)
		return result;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			uchar tmp = { 0 };
			float sumone = 0;
			for (int k = -int(round(sigma * 2)); k <= int(round(sigma * 2)); k++)
			{
				for (int l = -int(round(sigma * 2)); l <= int(round(sigma * 2)); l++)
				{
					if (i + k >= h || j + l >= w || i + k < 0 || j + l < 0) {
						continue;
					}
					tmp = tmp + Gaussian(k, l, sigma)*frame.at<uchar>(min(max(0, i + k), h - 1), min(max(j + l, 0), w - 1));
					sumone = sumone + Gaussian(k, l, sigma);
				}
			}
			result.at<uchar>(i, j) = uchar(round(tmp / sumone));
		}
	}
	return result;
}


Mat Resize(Mat frame, float ratio, int w, int h)
{
	int row = int(h * ratio);
	int col = int(w * ratio);
	Mat downsampled(row, col, CV_8UC1);
	for (int i = 0; i < row - 1; i++)
	{
		for (int j = 0; j < col - 1; j++)
		{
			downsampled.at<uchar>(i, j) = frame.at<uchar>(int((1 / ratio)*i), int((1 / ratio)*j));
		}
	}
	return downsampled;
}

void MakeGaussians(Mat** Octaves, Mat frame, int level, int size, int w, int h)
{
	int w_d = w;
	int h_d = h;
	Mat downsampled = frame.clone();
	float sigma = 0;
	for (int i = 0; i < level; i++) //다른 octave : size down
	{
		sigma = 0.5;
		float sigma_ratio = 0.5;//파라미터
		float ratio = 0.5;
		if (i == 0)
			ratio = 1;
		downsampled = Resize(downsampled, ratio, w_d, h_d);
		Octaves[i][0] = Mat(downsampled.clone());
		w_d = int(w_d*ratio);
		h_d = int(h_d*ratio);
		for (int j = 1; j < size; j++) //같은octave: sigma값 변경
		{
			Octaves[i][j] = GaussianFilter(downsampled, w_d, h_d, sigma);
			sigma = sigma + sigma_ratio;//* sigma_ratio;

		}
	}

}
void MakeDoG(Mat** DoG, Mat**Octaves, int level, int size)
{
	for (int i = 0; i < level; i++)
	{
		for (int j = 0; j < size - 1; j++)
		{
			DoG[i][j] = Octaves[i][j] - Octaves[i][j + 1]; // level, size
		}
	}
}

Mat finddelta(Mat** DoG, int x, int y, int lv, int s, int w, int h) //size: sigma의 size임. /candidate keypoint에대해 적용.
{
	Mat H = Mat::zeros(3, 3, CV_32FC1);
	//Dxx(0,0) Dxy(0,1) Dxs(0,2)
	//Dyx(1,0) Dyy(1,1) Dys(1,2)
	//Dsx(2,0) Dsy(2,1) Dss(2,2)
	H.at<float>(0, 0) =
		DoG[lv][s].at<uchar>(y, x + 1) - 2 * DoG[lv][s].at<uchar>(y, x) +
		DoG[lv][s].at<uchar>(y, x - 1);
	H.at<float>(0, 1) =
		((DoG[lv][s].at<uchar>(y + 1, x + 1) - DoG[lv][s].at<uchar>(y + 1, x - 1)) / 2
			- (DoG[lv][s].at<uchar>(y - 1, x + 1) - DoG[lv][s].at<uchar>(y - 1, x - 1)) / 2) / 2;
	H.at<float>(0, 2) = ((DoG[lv][s + 1].at<uchar>(y, x + 1) - DoG[lv][s + 1].at<uchar>(y, x - 1)) / 2
		- (DoG[lv][s - 1].at<uchar>(y, x + 1) - DoG[lv][s - 1].at<uchar>(y, x - 1)) / 2) / 2;
	//
	H.at<float>(1, 0) = H.at<float>(0, 1) = ((DoG[lv][s].at<uchar>(y + 1, x + 1) - DoG[lv][s].at<uchar>(y - 1, x + 1)) / 2
		- (DoG[lv][s].at<uchar>(y + 1, x - 1) - DoG[lv][s].at<uchar>(y - 1, x - 1)) / 2) / 2;
	H.at<float>(1, 1) = DoG[lv][s].at<uchar>(y + 1, x) - 2 * DoG[lv][s].at<uchar>(y, x) +
		DoG[lv][s].at<uchar>(y - 1, x);
	H.at<float>(1, 2) = ((DoG[lv][s + 1].at<uchar>(y + 1, x) - DoG[lv][s + 1].at<uchar>(y - 1, x)) / 2
		- (DoG[lv][s - 1].at<uchar>(y + 1, x) - DoG[lv][s - 1].at<uchar>(y - 1, x)) / 2) / 2;
	//
	H.at<float>(2, 0) = ((DoG[lv][s + 1].at<uchar>(y - 1, x) - DoG[lv][s - 1].at<uchar>(y, x + 1)) / 2
		- (DoG[lv][s + 1].at<uchar>(y, x - 1) - DoG[lv][s - 1].at<uchar>(y, x - 1)) / 2) / 2;
	H.at<float>(2, 1) = ((DoG[lv][s + 1].at<uchar>(y + 1, x) - DoG[lv][s - 1].at<uchar>(y + 1, x)) / 2
		- (DoG[lv][s + 1].at<uchar>(y - 1, x) - DoG[lv][s - 1].at<uchar>(y - 1, x)) / 2) / 2;
	H.at<float>(2, 2) = DoG[lv][s + 1].at<uchar>(y, x) - 2 * DoG[lv][s].at<uchar>(y, x) +
		DoG[lv][s - 1].at<uchar>(y, x);

	Mat Dmat = Mat::zeros(1, 3, CV_32FC1);
	Dmat.at<float>(0, 0) = (DoG[lv][s].at<uchar>(y, x + 1) - DoG[lv][s].at<uchar>(y, x - 1)) / 2;
	Dmat.at<float>(0, 1) = (DoG[lv][s].at<uchar>(y + 1, x) - DoG[lv][s].at<uchar>(y - 1, x)) / 2;
	Dmat.at<float>(0, 2) = (DoG[lv][s + 1].at<uchar>(y, x) - DoG[lv][s - 1].at<uchar>(y, x)) / 2;

	Mat delta = -H.inv()*Dmat.t();
	return delta;
}
void FindLocalExtrema(Mat** DoG, int level, int size, int w, int h, int** idx)
{
	idx[0][0] = 0;
	int cnt = idx[0][0];
	Mat Mask = Mat::zeros(h, w, CV_8UC1);
	float diffthreshold = 20;//파라미터
	float ratio = 0.5;
	int w_d = w;
	int h_d = h;
	for (int lv = 0; lv < level; lv++)
	{
		for (int sz = 1; sz < size - 1; sz++)//sigma값 변화
		{
			for (int i = 1; i < h_d - 1; i++)
			{
				for (int j = 1; j < w_d - 1; j++)
				{
					int j_tmp = j;
					int i_tmp = i;
					int s_tmp = sz;
					int flag = 0;
					int maximum = -1;
					int sum = 0;
					for (int k = -1; k <= 1; k++)//y
					{
						for (int l = -1; l <= 1; l++)//x
						{
							for (int m = -1; m <= 1; m++)//m:옥타브
							{
								if (k == 0 and l == 0 and m == 0)
									continue;
								if (maximum <= DoG[lv][sz + m].at<uchar>(i + k, j + l))
								{
									maximum = DoG[lv][sz + m].at<uchar>(i + k, j + l);
								}
								sum += DoG[lv][sz + m].at<uchar>(i + k, j + l);
							}
						}
					}
					sum = sum / 27;
					int cur = DoG[lv][sz].at<uchar>(i, j);
					if ((cur > maximum) && abs(cur - sum) >= 1)
					{
						Mat delta = finddelta(DoG, j_tmp, i_tmp, lv, s_tmp, w_d, h_d);
						if (!(abs(delta.at<float>(0, 0)) < 0.5&& abs(delta.at<float>(1, 0)) < 0.5 && abs(delta.at<float>(2, 0)) < 0.5))
						{
							int seriescnt = 0;
							while (!(abs(delta.at<float>(0, 0)) < 0.5 && abs(delta.at<float>(1, 0)) < 0.5 && abs(delta.at<float>(2, 0)) < 0.5) && seriescnt <= 4)
							{
								delta = finddelta(DoG, j_tmp, i_tmp, lv, s_tmp, w_d, h_d);
								j_tmp += int(round(delta.at<float>(0, 0)));//전개 h값 더해줌
								i_tmp += int(round(delta.at<float>(1, 0)));
								s_tmp += int(round(delta.at<float>(2, 0)));

								if (i_tmp >= h_d - 3 || j_tmp >= w_d - 3 || i_tmp <= 3 || j_tmp < 3 || s_tmp < 1 || s_tmp >2 || seriescnt >= 3)
								{
									flag = 1;
									j_tmp = j;
									i_tmp = i;
									s_tmp = sz;
									break;
								}
								seriescnt++;

							}
						}
						cur = DoG[lv][s_tmp].at<uchar>(i_tmp, j_tmp);

						float dx = DoG[lv][s_tmp].at<uchar>(i_tmp, j_tmp + 1) - 2 * cur + DoG[lv][s_tmp].at<uchar>(i_tmp, j_tmp - 1);
						float dy = DoG[lv][s_tmp].at<uchar>(i_tmp + 1, j_tmp) - 2 * cur + DoG[lv][s_tmp].at<uchar>(i_tmp - 1, j_tmp);
						float diag_l = DoG[lv][s_tmp].at<uchar>(i_tmp + 1, j_tmp + 1) - 2 * cur + DoG[lv][s_tmp].at<uchar>(i_tmp - 1, j_tmp + 1);
						float diag_r = DoG[lv][s_tmp].at<uchar>(i_tmp - 1, j_tmp - 1) - 2 * cur + DoG[lv][s_tmp].at<uchar>(i_tmp + 1, j_tmp - 1);


						Mat Dmat = Mat::zeros(1, 3, CV_32FC1);
						Dmat.at<float>(0, 0) = (DoG[lv][s_tmp].at<uchar>(i_tmp, j_tmp + 1) - DoG[lv][s_tmp].at<uchar>(i_tmp, j_tmp - 1)) / 2;
						Dmat.at<float>(0, 1) = (DoG[lv][s_tmp].at<uchar>(i_tmp + 1, j_tmp) - DoG[lv][s_tmp].at<uchar>(i_tmp - 1, j_tmp)) / 2;
						Dmat.at<float>(0, 2) = (DoG[lv][s_tmp + 1].at<uchar>(i_tmp, j_tmp) - DoG[lv][s_tmp - 1].at<uchar>(i_tmp, j_tmp)) / 2;

						float contrast = Dmat.at<float>(0, 0) * delta.at<float>(0, 0) + Dmat.at<float>(0, 1) * delta.at<float>(1, 0) + Dmat.at<float>(0, 2) * delta.at<float>(2, 0);
						if (abs(contrast) < 1)
							flag = 1;
						if (abs(dx) < diffthreshold || abs(dy) < diffthreshold || abs(diag_r) < diffthreshold || abs(diag_l) < diffthreshold)
							flag = 1;

						if (flag == 0)
						{

							if (Mask.at<uchar>(i_tmp*int(pow(1 / ratio, lv)), j_tmp*int(pow(1 / ratio, lv))) == 0)
								cnt++;
							if (Mask.at<uchar>(i_tmp*int(pow(1 / ratio, lv)), j_tmp*int(pow(1 / ratio, lv))) < DoG[lv][s_tmp].at<uchar>(i_tmp, j_tmp))
							{
								Mask.at<uchar>(i_tmp*int(pow(1 / ratio, lv)), j_tmp*int(pow(1 / ratio, lv))) = DoG[lv][s_tmp].at<uchar>(i_tmp, j_tmp); //1/ratio x 1/ratio만큼 downscaling이므로 원이미지에서는 1/ratio의 lv제곱.7

								idx[cnt][0] = i_tmp * int(pow(1 / ratio, lv));
								idx[cnt][1] = j_tmp * int(pow(1 / ratio, lv));
								idx[cnt][2] = s_tmp;
							}

						}
					}
				}
			}
		}
		w_d = int(w_d*ratio);
		h_d = int(h_d*ratio);
	}
	Mask = Mask / 2;
	idx[0][0] = cnt;

}


void motionExtraction(Mat frame1, Mat frame2, int cnt, int** idx, int** idx_aft, int h, int w)
{
	int n = 5; // 블럭크기
	int range = int(n / 2);
	Mat t[2];
	t[0] = frame1;
	t[1] = frame2;

	for (int i = 1; i <= cnt; i++)
	{
		Mat A = Mat::zeros((n*n), 2, CV_32FC1);
		Mat b = Mat::zeros((n*n), 1, CV_32FC1);
		int x = idx[i][1];
		int y = idx[i][0];
		if (x + range + 1 >= w || x - range < 0 || y + range + 1 >= h || y - range < 0)
		{
			idx_aft[i][0] = 0;
			idx_aft[i][1] = 0;
			continue;
		}
		for (int r = y - range; r <= y + range; r++)
		{
			for (int c = x - range; c <= x + range; c++)
			{
				A.at<float>(
					(r - (y - range))*n + (c - (x - range)), 0) = t[0].at<uchar>(r + 1, c) - t[0].at<uchar>(r, c);
				A.at<float>(
					(r - (y - range))*n + (c - (x - range)), 1) = t[0].at<uchar>(r, c + 1) - t[0].at<uchar>(r, c);
				b.at<float>((r - (y - range))*n + (c - (x - range)), 0) = -(t[1].at<uchar>(r, c) - t[0].at<uchar>(r, c));
			}
		}
		Mat v = (A.t()*A).inv()*A.t()*b;
		cout << v << endl;
		idx_aft[i][0] = int(round(v.at<float>(0, 0) * 2));//y
		idx_aft[i][1] = int(round(v.at<float>(1, 0) * 2));//x
	}
}