#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include "opencv2/opencv.hpp"

#include <iostream>

#include <stdio.h>

#include <math.h>

#include <time.h>

#include <string>

#include <algorithm>





////// I N F O R M A T I O N  -   C U D A D E V I C E /////////////////



//total amount of global memory : 2048 MB (21 4748 3648)

//L2 Cache Size 52 4288 byte

//Maximum Texture Size: 16384,16384 , 2048 layers

//constant memory : 65536 bytes

//shared memory per block : 49152 bytes

//registers available per block: 65536

// warp size : 32

//maximum num of thread per block 1024

//max dimension size of thread block : (1024,1024,64)

//max dimension size of grid size : (21 4748 3647, 65535, 6535)

//max memory pitch : 21 4748 3647 byte




using namespace cv;
using namespace std;


int diff_t = 5;
int diff_c = 5;
int blocksize = 129;

int keycnt = 0;

int startflag = 1;




int w = 1920;//960;// 1920;

int h = 1080;//540;//1080;

////////////////C U D A///////////////////////////////////////////////////////////////////////////////



#define TILE_WIDTH 32



////////////////////////////////////////////////////////////////////
__global__ void FrameCopy(uchar* copy, uchar* frame, int w, int h);
__global__ void ResizeGPU(uchar* output, uchar* frame, float ratio, int w, int h);
__global__ void GaussianFilterGPU(uchar* output, uchar* frame, int w, int h, float* filter);
__global__ void Derivate(uchar* DoG, uchar* Octave1, uchar* Octave2, int w, int h);

__device__ void finddelta(uchar* DoG_ma, uchar* DoG, uchar* DoG_pl, int x, int y, int w, int h_d, float* delta);
__global__ void FindLocalExtrema(uchar* Mask, uchar* DoG_ma, uchar* DoG, uchar* DoG_pl, int lv, int h, int w, float diff_t, float diff_c);
__global__ void setZero(uchar* Mask, int w, int h);
__global__ void LucasKanade(uchar* f1, uchar* f2, int xp, int yp, float* A, float* b, int n, int range, int w);

float Gaussian(int u, int v, float sigma);//basic
uchar* Resize(uchar* frame, float ratio, int w, int h);//basic
void MakeGaussians(uchar*** Octaves, uchar* frame_gpu, int level, int size, int w, int h, float** filter);
void MakeDoG(uchar*** DoG, uchar***Octaves, int level, int size);
void FeatureExtracion(uchar*** DoG, int level, int size, int w, int h, int** idx, float diff_t, float diff_c);
void motionExtraction(uchar* frame1, uchar* frame2, int cnt, int** idx, int** idx_aft, int h, int w);

/////////////////////////////////////////////////////////////////////////////////////////////////



uchar *frame_gpu;
uchar *frame_pre_gpu;
uchar*** octave_gpu;
uchar*** dog_gpu;

int** idx;
int** idx_aft;


int main(int argc, char **argv)
{

	//1.input
	int cnt = 0;
	float sigma = 0.5;
	float filter[5][9 * 9];
	float* filter_gpu[5];
	for (int i = 0; i < 5; i++)
	{
		for (int a = 0; a < 9; a++)
		{
			for (int b = 0; b < 9; b++)
			{
				filter[i][a * 9 + b] = Gaussian(a - 4, b - 4, sigma);
			}
		}
		sigma = sigma + 1;
	}
	for (int i = 0; i < 5; i++)
	{
		cudaMalloc((void**)&filter_gpu[i], 9 * 9 * sizeof(float));
		cudaMemcpy(filter_gpu[i], filter[i], 9 * 9 * sizeof(float), cudaMemcpyHostToDevice);
	}

	cudaSetDevice(0);



	VideoCapture cap("sea2.mp4");



	Mat frame1;
	Mat frame_g;
	Mat frame_g_pre;


	while (cap.isOpened())
	{
		cap.read(frame1);

		if (frame1.empty()) {

			cerr << "empty.\n";

			break;

		}
		cv::cvtColor(frame1, frame_g_pre, COLOR_RGB2GRAY);
		break;
	} 


	clock_t assign_pre = clock();
	octave_gpu = new uchar**[4];
	for (int i = 0; i < 4; i++)
	{
		octave_gpu[i] = new uchar*[5];
	}
	dog_gpu = new uchar**[4];
	for (int i = 0; i < 4; i++)
	{
		dog_gpu[i] = new uchar*[4];
	}

	float r = 0.5;
	int w_tmp = w;
	int h_tmp = h;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			cudaMalloc((void**)&octave_gpu[i][j], w_tmp*h_tmp * sizeof(uchar)); //엑세스 위반 오류 ****
			if (j < 4)
				cudaMalloc((void**)&dog_gpu[i][j], w_tmp*h_tmp * sizeof(uchar));
		}
		w_tmp = int(w_tmp*r);
		h_tmp = int(h_tmp*r);
	}

	clock_t assign_cur = clock();
	cout << "memory assignment : " << assign_cur - assign_pre << endl;




	clock_t idxasspre = clock();
	idx = new int*[1920 * 1080];
	for (int i = 0; i < 1920 * 1080; i++)
	{
		idx[i] = new int[2];
	}
	idx_aft = new int*[1920 * 1080];
	for (int i = 0; i < 1920 * 1080; i++)
	{
		idx_aft[i] = new int[2];
	}
	clock_t idxasscur = clock();








	cudaMalloc((void **)&frame_gpu, w * h * sizeof(uchar)); //원본프레임 GPU
	cudaMalloc((void **)&frame_pre_gpu, w * h * sizeof(uchar)); //이전프레임 GPU

	while (cap.isOpened())

	{

		cap.read(frame1);

		if (frame1.empty()) {

			cerr << "empty.\n";

			break;

		}
		clock_t pre = clock();


		cv::resize(frame1, frame1, cv::Size(int(w), int(h)), cv::INTER_LINEAR);
		cv::cvtColor(frame1, frame_g, COLOR_RGB2GRAY);


		clock_t mem_pre = clock();

		uchar* array = frame_g.data;
		cudaMemcpy(frame_gpu, array, w * h * sizeof(uchar), cudaMemcpyHostToDevice);//앞에가 받을거 주소host(cpu) dev(cuda device주소)
		clock_t mem_cur = clock();




		cout << "frame to gpu : " << mem_cur - mem_pre << endl;

		if (startflag == 1)
		{




			clock_t gaupre = clock();
			MakeGaussians(octave_gpu, frame_gpu, 4, 5, w, h, filter_gpu);//level:4 size:5
			clock_t gaucur = clock();
			cout << "makegaussian : " << gaucur - gaupre << endl;

			MakeDoG(dog_gpu, octave_gpu, 4, 5);

			/////////////////////////////////////////////////////////////////////////////////////////////

			r = 0.5;
			w_tmp = w;
			h_tmp = h;

			cout << "idx assignment : " << idxasscur - idxasspre << endl;



			clock_t fexpre = clock();
			FeatureExtracion(dog_gpu, 4, 4, w, h, idx, diff_t, diff_c);
			clock_t fexcur = clock();
			cout << "featureExtraction : " << fexcur - fexpre << endl;
			cnt = idx[0][0];
			for (int i = 0; i < cnt; i++)
			{
				circle(frame1, Point(idx[i][0], idx[i][1]), 3, Scalar(0, 255, 255));
			}


		}

		else if (startflag == 0)
		{
			clock_t mepre = clock();
			motionExtraction(frame_pre_gpu, frame_gpu, cnt, idx, idx_aft, h, w);
			clock_t mecur = clock();
			cout << "motion extraction : " << mecur - mepre << endl;


			for (int i = 1; i < cnt; i++)
			{
				idx_aft[i][0] = min(max(0, int(idx[i][0]) + int(idx_aft[i][1])), w - 1);
				idx_aft[i][1] = min(max(int(idx[i][1]) + int(idx_aft[i][1]), 0), h - 1);
				circle(frame1, Point(idx_aft[i][0], idx_aft[i][1]), 3, Scalar(255, 0, 0));
				circle(frame1, Point(idx[i][0], idx[i][1]), 3, Scalar(0, 255, 0));
				idx[i][0] = idx_aft[i][0];
				idx[i][1] = idx_aft[i][1];
			}


		}

		clock_t cur = clock();
		clock_t sec = cur - pre;
		float fps = (sec) / double(CLOCKS_PER_SEC);
		string s = to_string(fps);
		putText(frame1, s, Point(0, 100), FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255));
		imshow("1", frame1);
		dim3 dimGrid((w - 1) / TILE_WIDTH + 1, (h - 1) / TILE_WIDTH + 1);
		dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
		FrameCopy << <dimGrid, dimBlock >> > (frame_pre_gpu, frame_gpu, w, h);



		if (waitKey(1) == 'x')

			break;

		else if (waitKey(1) == 'z')

		{

			startflag = 1;

			diff_t = 10;

			diff_c = 10;

			blocksize = 33;

		}
		else if (waitKey(1) == 'c')

		{

			startflag = 1;

			diff_t = 15;

			diff_c = 15;

			blocksize = 33;

		}

		cout << "time : " << cur - pre << endl;

	}



	return 0;

}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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
__global__ void FrameCopy(uchar* copy, uchar* frame, int w, int h)
{
	int i = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x*TILE_WIDTH + threadIdx.x;
	if (i < h && j < w)
	{
		copy[i*w + j] = frame[i*w + j];
	}
}
__global__ void ResizeGPU(uchar* output, uchar* frame, float ratio, int w, int h)
{
	int i = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x*TILE_WIDTH + threadIdx.x;
	if (i < h && j < w)
	{
		output[i*w + j] = frame[int(1 / ratio)*int(1 / ratio)*i*w + int(1 / ratio)*j];
	}
}

__global__ void GaussianFilterGPU(uchar* output, uchar* frame, int w, int h, float* filter)
{
	int i = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x*TILE_WIDTH + threadIdx.x;
	if (i < h && j < w)
	{
		//uchar tmp = { 0 };
		float tmp = 0;

		float sumone = 0;

		for (int k = -4; k <= 4; k++)//커널사이즈 : 9*9로. 펼치기할것.

		{

			for (int l = -int(4); l <= int(4); l++)
			{

				tmp = tmp + filter[(k + 4) * 9 + l + 4] * frame[max(0, min(h - 1, (i + k)))*w + max(0, min((j + l), w - 1))];
				//방1) max & min 대신 if문 안쓰는 방법
				//방2) shared memory 이용

				sumone = sumone + filter[(k + 4) * 9 + l + 4];

			}

		}

		output[i*w + j] = uchar(round(tmp / sumone));
	}
}
__global__ void Derivate(uchar* DoG, uchar* Octave1, uchar* Octave2, int w, int h)
{
	int i = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x*TILE_WIDTH + threadIdx.x;
	if (i < h && j < w)
	{
		DoG[i*w + j] = uchar((min(255, max(0, int(Octave1[i*w + j]) - int(Octave2[i*w + j]))))); // level, size
		//DoG[i*w + j] = uchar(abs(int(Octave1[i*w + j]) - int(Octave2[i*w + j]))); // level, size
		/*DoG[i*w + j] = min(255, max(0, Octave1[i*w + j] - Octave2[i*w + j]));*/
		//DoG[i*w + j] = min(255, max(0, -Octave1[i*w + j] + Octave2[i*w + j]));
	}
}

__device__ void finddelta(uchar* DoG_ma, uchar* DoG, uchar* DoG_pl, int x, int y, int w, int h_d, float* delta) //size: sigma의 size임. /candidate keypoint에대해 적용.
{
	float H[3][3] = { 0 };
	//Dxx(0,0) Dxy(0,1) Dxs(0,2)
	//Dyx(1,0) Dyy(1,1) Dys(1,2)
	//Dsx(2,0) Dsy(2,1) Dss(2,2)
	H[0][0] = DoG[y*w + x + 1] - 2 * DoG[y*w + x] + DoG[y*w + x - 1];
	H[0][1] = ((DoG[(y + 1)*w + x + 1] - DoG[(y + 1)*w + x - 1]) / 2 - (DoG[(y - 1)*w + x + 1] - DoG[(y - 1)*w + x - 1]) / 2) / 2;
	H[0][2] = ((DoG_pl[y*w + x + 1] - DoG_pl[y*w + x - 1]) / 2 - (DoG_ma[y*w + x + 1] - DoG_ma[y*w + x - 1]) / 2) / 2;

	H[1][0] = ((DoG[(y + 1)*w + x + 1] - DoG[(y - 1)*w + x + 1]) / 2 - (DoG[(y + 1)*w + x - 1] - DoG[(y - 1)*w + x - 1]) / 2) / 2;
	H[1][1] = DoG[(y + 1)*w + x] - 2 * DoG[y*w + x] + DoG[(y - 1)*w + x];
	H[1][2] = ((DoG_pl[(y + 1)*w + x] - DoG_pl[(y - 1)*w + x]) / 2 - (DoG_ma[(y + 1)*w + x] - DoG_ma[(y - 1)*w + x]) / 2) / 2;

	H[2][0] = ((DoG_pl[(y)*w + x + 1] - DoG_ma[y*w + x + 1]) / 2 - (DoG_pl[y*w + x - 1] - DoG_ma[y * w + x - 1]) / 2) / 2; //다른코드에서 이부분 잘못되잇음.
	H[2][1] = ((DoG_pl[(y + 1)*w + x] - DoG_ma[(y + 1)*w + x]) / 2 - (DoG_pl[(y - 1)*w + x] - DoG_ma[(y - 1)*w + x]) / 2) / 2;
	H[2][2] = DoG_pl[y*w + x] - 2 * DoG[y*w + x] + DoG_ma[y*w + x];

	float D[3];
	D[0] = (DoG[y*w + x + 1] - DoG[y*w + x - 1]) / 2;
	D[1] = (DoG[(y + 1)*w + x] - DoG[(y - 1)*w + x]) / 2;
	D[2] = (DoG_pl[y*w + x] - DoG_ma[y*w + x]) / 2;

	float a = H[0][0];
	float b = H[0][1];
	float c = H[0][2];
	float d = H[1][0];
	float e = H[1][1];
	float f = H[1][2];
	float g = H[2][0];
	float h = H[2][1];
	float i = H[2][2];

	float det = 1 / (a*e*i - a * f*h - b * f*g + c * d*h - c * e*g);
	float Hinv[3][3];
	Hinv[0][0] = (e*i - f * h)*det;
	Hinv[0][1] = (c*h - b * i)*det;
	Hinv[0][2] = (b*f - c * e)*det;
	Hinv[1][0] = (f*g - d * i)*det;
	Hinv[1][1] = (a* i - c * g)*det;
	Hinv[1][2] = (c*d - a * f)*det;
	Hinv[2][0] = (d*h - e * g)*det;
	Hinv[2][1] = (b*g - a * h)*det;
	Hinv[2][2] = (a*e - b * d)*det;


	delta[0] = Hinv[0][0] * D[0] + Hinv[0][1] * D[1] + Hinv[0][2] * D[2];
	delta[1] = Hinv[1][0] * D[0] + Hinv[1][1] * D[1] + Hinv[1][2] * D[2];
	delta[2] = Hinv[2][0] * D[0] + Hinv[2][1] * D[1] + Hinv[2][2] * D[2];

}



__global__ void FindLocalExtrema(uchar* Mask, uchar* DoG_ma, uchar* DoG, uchar* DoG_pl, int lv, int h, int w, float diff_t, float diff_c)
{


	int i = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x*TILE_WIDTH + threadIdx.x;
	int maximum = -9999;
	if (i < h - 1 && j < w - 1 && i>0 && j>0)
	{
		int flag_localextrema = 1;
		int flag_feature = 0;


		for (int k = -1; k <= 1; k++)//y
		{
			for (int l = -1; l <= 1; l++)//x
			{
				if (k == 0 && l == 0)
				{
					if (maximum <= int(DoG_ma[(i + k)*w + (j + l)]))
					{
						maximum = DoG_ma[(i + k)*w + j];
					}
					if (maximum <= int(DoG_pl[(i + k)*w + (j + l)]))
					{
						maximum = DoG_pl[(i + k)*w + j];
					}
				}
				else
				{
					if (maximum <= int(DoG[(i + k)*w + (j + l)]))
					{
						maximum = DoG[(i + k)*w + j];
					}
					if (maximum <= int(DoG_ma[(i + k)*w + (j + l)]))
					{
						maximum = DoG_ma[(i + k)*w + j];
					}
					if (maximum <= int(DoG_pl[(i + k)*w + (j + l)]))
					{
						maximum = DoG_pl[(i + k)*w + j];
					}
				}
			}

		}

		uchar cur = DoG[i*w + j];
		if (cur == maximum)
			flag_localextrema = 1;
		if (flag_localextrema == 1 && cur >= 1)
		{
			flag_feature = 1;
			float delta[3];

			finddelta(DoG_ma, DoG, DoG_pl, j, i, w, h, delta);



			float dx = DoG[i*w + j + 1] - 2 * cur + DoG[i*w + j - 1];
			float dy = DoG[(i + 1)*w + j] - 2 * cur + DoG[(i - 1)*w + j];
			float diag_l = DoG[(i + 1)*w + (j + 1)] - 2 * cur + DoG[(i - 1)*w + (j - 1)]; //수정함
			float diag_r = DoG[(i + 1)*w + (j - 1)] - 2 * cur + DoG[(i - 1)*w + (j + 1)]; //수정함


			float Dmat[3];
			Dmat[0] = (DoG[i*w + j + 1] - DoG[i*w + j - 1]) / 2;
			Dmat[1] = (DoG[(i + 1)*w + j] - DoG[(i - 1)*w + j]) / 2;
			Dmat[2] = (DoG_pl[i*w + j] - DoG_ma[i*w + j]) / 2;

			float contrast = Dmat[0] * delta[0] + Dmat[1] * delta[1] + Dmat[2] * delta[2];

			if (abs(dx) < diff_t || abs(dy) < diff_t || abs(diag_r) < diff_t || abs(diag_l) < diff_t || abs(contrast) < diff_c)
				flag_feature = 0;

			if (flag_feature == 1)
			{
				int idx_x = j * int(pow(1 / 0.5, lv));//ratio:0.5
				int idx_y = i * int(pow(1 / 0.5, lv));

				//Mask[idx_y * 1920 + idx_x] = ((DoG[i*w + j] > Mask[idx_y * 1920 + idx_x]) ? DoG[i*w + j] : Mask[idx_y * 1920 + idx_x]);
				Mask[idx_y * 1920 + idx_x] = 255;
			}
		}

	}
}
__global__ void LucasKanade(uchar* f1, uchar* f2, int xp, int yp, float* A, float* b, int n, int range, int w)
{
	int i = blockIdx.y*TILE_WIDTH + threadIdx.y; //0~n
	int j = blockIdx.x*TILE_WIDTH + threadIdx.x;
	if (i < n && j < n)
	{
		A[2 * (i * n + j)] = float(
			int(f1[(yp + i - range + 1)*w + (xp + j - range)])
			- int(f1[(yp + i - range)*w + (xp + j - range)])
			);
		A[2 * (i * n + j) + 1] = float(
			int(f1[(yp + i - range)*w + (xp + j - range + 1)])
			- int(f1[(yp + i - range)*w + (xp + j - range)])
			);
		b[i*n + j] = -1 * float(
			int((f2[(yp + i - range)*w + (xp + j - range)])
				- int(f1[(yp + i - range)*w + (xp + j - range)]))
			);
	}
}

__global__ void setZero(uchar* Mask, int w, int h)
{

	int i = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x*TILE_WIDTH + threadIdx.x;
	if (i < h && j < w)
	{
		Mask[i*w + j] = 0;
	}
}





void MakeDoG(uchar*** DoG, uchar***Octaves, int level, int size)
{
	int w_d = w;
	int h_d = h;
	for (int i = 0; i < level; i++)
	{
		dim3 dimGrid((w_d - 1) / TILE_WIDTH + 1, (h_d - 1) / TILE_WIDTH + 1);
		dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
		for (int j = 0; j < size - 1; j++)
		{
			Derivate << <dimGrid, dimBlock >> > (DoG[i][j], Octaves[i][j], Octaves[i][j + 1], w_d, h_d);
			cudaDeviceSynchronize();
		}
		w_d = int(w_d * 0.5);
		h_d = int(h_d * 0.5);
	}
}

void MakeGaussians(uchar*** Octaves, uchar* frame_gpu, int level, int size, int w, int h, float** filter)

{

	//cout << "Enter" << endl;
	int w_d = w;

	int h_d = h;

	float sigma = 0;

	float ratio = 1;
	for (int i = 0; i < level; i++) //다른 octave : size down

	{
		sigma = 0.5;

		float sigma_ratio = 1;//파라미터

		//float ratio = 0.5;
		w_d = int(w*ratio);
		h_d = int(h*ratio);
		dim3 dimGrid((w_d - 1) / TILE_WIDTH + 1, (h_d - 1) / TILE_WIDTH + 1);
		dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

		if (i == 0)
		{
			cudaMemcpy(Octaves[i][0], frame_gpu, w * h * sizeof(uchar), cudaMemcpyHostToDevice);

		}
		else if (i > 0)
		{
			clock_t respre = clock();
			ResizeGPU << <dimGrid, dimBlock >> > (Octaves[i][0], frame_gpu, ratio, w_d, h_d);
			cudaDeviceSynchronize();
			clock_t rescur = clock();
			cout << "i : " << i << ", resize : " << rescur - respre << endl;


		}



		for (int j = 1; j < size; j++) //같은octave: sigma값 변경

		{


			clock_t filterpre = clock();
			GaussianFilterGPU << <dimGrid, dimBlock >> > (Octaves[i][j], Octaves[i][0], w_d, h_d, filter[j]); //GAUSSIAN FILTER 거치면 RESIZE까지 영향감.?
			cudaDeviceSynchronize();
			clock_t filtercur = clock();
			cout << "i : " << i << "j : " << j << ", filter : " << filtercur - filterpre << endl;



			sigma = sigma + sigma_ratio;//* sigma_ratio;




		}
		ratio = ratio * 0.5;
	}

}


void FeatureExtracion(uchar*** DoG, int level, int size, int w, int h, int** idx, float diff_t, float diff_c)//output : index ?
{
	idx[0][0] = 0;
	int cnt = 0;
	uchar* Mask;
	uchar* Mask_cpu = new uchar[1920 * 1080];
	cudaMalloc((void **)&Mask, 1920 * 1080 * sizeof(uchar));
	dim3 dimGrid((w - 1) / TILE_WIDTH + 1, (h - 1) / TILE_WIDTH + 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	setZero << <dimGrid, dimBlock >> > (Mask, w, h);
	cudaDeviceSynchronize();
	float ratio = 0.5;
	int w_d = w;
	int h_d = h;

	for (int lv = 0; lv < level; lv++)
	{
		for (int sz = 1; sz < size - 1; sz++)//sigma값 변화
		{

			dim3 dimGrid((w_d - 1) / TILE_WIDTH + 1, (h_d - 1) / TILE_WIDTH + 1);
			dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

			FindLocalExtrema << <dimGrid, dimBlock >> > (Mask, DoG[lv][sz - 1], DoG[lv][sz], DoG[lv][sz + 1], lv, h_d, w_d, diff_t, diff_c);
			cudaDeviceSynchronize();
		}
		w_d = int(w_d*ratio);
		h_d = int(h_d*ratio);
	}
	cudaMemcpy(Mask_cpu, Mask, 1920 * 1080 * sizeof(uchar), cudaMemcpyDeviceToHost);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (Mask_cpu[i*w + j] != 0)
			{
				cnt++;
				idx[cnt][0] = j;
				idx[cnt][1] = i;
			}
		}
	}
	idx[0][0] = cnt;
}

void motionExtraction(uchar* frame1, uchar* frame2, int cnt, int** idx, int** idx_aft, int h, int w)
{
	int n = blocksize; // 블럭크기
	int range = int(n / 2);

	for (int i = 1; i <= cnt; i++)
	{
		n = blocksize;
		range = int(n / 2);
		int x = idx[i][1];
		int y = idx[i][0];
		if (x + 5 >= w || x < 5 || y + 5 >= h || y < 5)
		{
			idx_aft[i][0] = 0;
			idx_aft[i][1] = 0;
			continue;
		}
		else if (x + range + 1 >= w || x - range < 0 || y + range + 1 >= h || y - range < 0)
		{
			n = 8;
			range = int(n / 2);
		}

		float* A_g;
		cudaMalloc((void**)&A_g, n*n * 2 * sizeof(float));
		float* b_g;
		cudaMalloc((void**)&b_g, n*n * sizeof(float));

		int tile = 8;
		dim3 dimGrid((n - 1) / tile + 1, (n - 1) / tile + 1);
		dim3 dimBlock(tile, tile);
		clock_t lucpre = clock();
		LucasKanade << <dimGrid, dimBlock >> > (frame1, frame2, x, y, A_g, b_g, n, range, w);

		clock_t luccur = clock();
		clock_t addpre = clock();
		float* tmp_A = new float[n*n * 2];
		float* tmp_b = new float[n*n];


		cudaMemcpy(tmp_A, A_g, n*n * 2 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(tmp_b, b_g, n*n * sizeof(float), cudaMemcpyDeviceToHost);

		cv::Mat A(n*n, 2, CV_32FC1);
		cv::Mat b(n*n, 1, CV_32FC1);
		memcpy(A.data, tmp_A, n*n * 2 * sizeof(float));
		memcpy(b.data, tmp_b, n*n * 1 * sizeof(float));

	
		Mat v = (A.t()*A).inv()*A.t()*b;



		////cout << v << endl;
		////cout << "roud test" << round(-0.6) << endl;
		idx_aft[i][0] = int(round(v.at<float>(1, 0) *1.0));//x
		idx_aft[i][1] = int(round(v.at<float>(0, 0) *1.0));//y
		cudaFree(A_g);
		cudaFree(b_g);
		//delete[] tmp_A;
		//delete[] tmp_b;
	}
}
