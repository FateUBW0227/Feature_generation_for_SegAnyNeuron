#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "voxel.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <curand.h>
#include <cudnn.h>
#include <atomic>


#define NUM_THREADS 256

extern "C"  void fast_iter(int level, Voxel_L &lrImage, Voxel_E &dirSet, Eigen::Tensor<float, 3> &ss,
	int ss_x, int ss_y, int ss_z, Voxel_E &predict, int type_id);

extern "C" void fast_iter2(int level, Voxel_L &lrImage, Eigen::Tensor<unsigned char, 3> &dirSet, Eigen::Tensor<float, 3> &ss,
	int ss_x, int ss_y, int ss_z, Voxel_E &predict,int type_id, double thred, double thred2);

extern "C" void calc_direction_id(std::vector<Eigen::Tensor<float, 3> >kernels, Voxel_L &lrImage, std::vector<Eigen::Tensor<float, 3> > &res,
	Eigen::Tensor<unsigned char, 3> &idx_res);

extern "C" void test_conv(std::vector<Eigen::Tensor<float, 3>> &oris, Eigen::Tensor<float, 3> &kernel,
	 std::vector<Eigen::Tensor<float, 3>> &res);

void calc_mean_std(unsigned short *lrImaged, int total_pixel_num, double mean, double std);

__global__ void initialize(float *src, float val, int num);

__global__ void initialize(unsigned short *src, unsigned short val, int num);

__global__ void conv_mult(float *ori, float *kernel, float *res, int dim_x, int dim_y, int dim_z, int kernel_dim, int total_pixel_num);

__global__ void conv_mult(unsigned short *ori, float *kernel, float *res, int dim_x, int dim_y, int dim_z, int kernel_dim, int total_pixel_num);

__global__ void
calc_feature_map(unsigned short *lrImage_d, unsigned char *dirset_d, float *ss_d, int dim_x, int dim_y, int dim_z, 
	unsigned char *predict, int total_pixel_num, int ss_x, int ss_y, int ss_z, int level, int type_id, double thred, double thred2);

__global__ void
calc_feature_map_with_meanstd(unsigned short *lrImage_d, unsigned char *dirset_d, float *ss_d, int dim_x, int dim_y, int dim_z,
	unsigned char *predict, int total_pixel_num, int ss_x, int ss_y, int ss_z, int level, int type_id, double thred, double thred2,
	double mean, double std);

__global__ void pad_img(unsigned short *ori, unsigned short *tar, int padding_x, int padding_y, int padding_z, 
	                    int total_pixel_num, int dim_x, int dim_y, int dim_z);

__global__ void obtain_ids(unsigned short *idx, float*res, int total_pixel_num, int kernel_num);

extern "C" void conv(Voxel_L &lrImage, std::vector<Eigen::Tensor<float, 3> > kernels, int dim_x, int dim_y, int dim_z, int kernel_dim, bool reverse_xyz, int padding,
	std::vector<Eigen::Tensor<float, 3> > &res, Eigen::Tensor<unsigned char, 3> &id_res);


__device__ void normalize(float *vec)
{
	double L = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	vec[0] = (vec[0]) / (L + 0.000000001);
	vec[1] = (vec[1]) / (L + 0.000000001);
	vec[2] = (vec[2]) / (L + 0.000000001);
}


__device__ void Coordinates(float *dir, float *res)
{
	float x = std::abs(dir[0]);
	float y = std::abs(dir[1]);
	float z = std::abs(dir[2]);
	int id = 0;
	if (x > y)
	{
		id = 1;
		if (z < y)
			id = 2;
	}
	else
	{
		if (z < x)
			id = 2;
	}
	int id0 = (id + 1) % 3;
	int id1 = (id + 2) % 3;
	res[0] = dir[0];
	res[1] = dir[1];
	res[2] = dir[2];

	res[id0 + 3] = dir[id1];
	res[id1 + 3] = - dir[id0];

	res[6] = res[1] * res[2 + 3] - res[2] * res[1 + 3];
	res[7] = -res[0] * res[2 + 3] + res[2] * res[0 + 3];
	res[8] = res[0] * res[1 + 3] - res[1] * res[0 + 3];
	normalize(res);
	normalize(res + 3);
	normalize(res + 6);
}

__device__ void set_value(float* adr, float val, int num)
{
	for (int i = 0; i < num; ++i)
	{
		adr[i] = val;
	}
}


__device__ void ImgLineSample(int *center, float *direction, int Num, unsigned short *lrImage_d, float *Points,
	int nx, int ny, int nz)
{
	double local_min = 999990000.0;
	double local_max = -10000.0;
	for (int i = 0; i < 2 * Num + 1; ++i)
	{
		int valid = 0;
		float p[3] = {center[0] + (i - Num) * direction[0], 
		              center[1] + (i - Num) * direction[1],
		              center[2] + (i - Num) * direction[2]};
		if (p[0] >= 0 && p[0] < nz - 1 &&
			p[1] >= 0 && p[1] < ny - 1 &&
			p[2] >= 0 && p[2] < nx - 1)
		{
			valid = 1;
		}
		if (valid)
		{
			int coor[3] = {std::round(p[0]), std::round(p[1]), std::round(p[2])};
			int I = lrImage_d[coor[0] + coor[1] * nx + coor[2] * nx * ny];
			if (I < local_min)
				local_min = I;
			if (I > local_max)
				local_max = I;
			Points[4 * i] = I;
			Points[4 * i + 1] = p[0];
			Points[4 * i + 2] = p[1];
			Points[4 * i + 3] = p[2];
		}
		else
		{
			Points[4 * i] = lrImage_d[center[0] + center[1] * nx + center[2] * nx * ny];
		}
	}
	/*for (int i = 0; i < 2 * Num + 1; ++i)
	{
		Points[4 * i] = (Points[4 * i] - local_min) / (local_max - local_min) * 2000;
	}*/
}


__device__ void ImgLineSampleSet(int *idx, float *res, const int Num, unsigned short *lrImage_d, int nx, int ny, int nz,
	float *Point0, float *Point1, float *Point2)
{
	set_value(Point0, 0.0, 4 * (2 * 4 + 1));
	set_value(Point1, 0.0, 4 * (2 * 4 + 1));
	set_value(Point2, 0.0, 4 * (2 * 4 + 1));
	ImgLineSample(idx, res, Num, lrImage_d, Point0, nx, ny, nz);
	ImgLineSample(idx, res + 3, Num, lrImage_d, Point1, nx, ny, nz);
	ImgLineSample(idx, res + 6, Num, lrImage_d, Point2, nx, ny, nz);
}

__device__ int Lineregiongrowing(float *Point, int index, float thred, int Point_num)
{
	int NUM = 0;
	for (int i = 0; i < Point_num - index; ++i)
	{
		if (Point[(i + index) * 4] > thred)
			NUM = NUM + 1;
		else
			break;
	}
	int NUM2 = 0;
	for (int i = 0; i < index; ++i)
	{
		if (Point[(index - i - 1) * 4] > thred)
			NUM2 = NUM2 + 1;
		else
			break;
	}
	return NUM + NUM2;
}


__device__ int Lineregiongrowing(float *Point, int index, float thred, float up_thred, int Point_num)
{
	int NUM = 0;
	for (int i = 0; i < Point_num - index; ++i)
	{
		if (Point[(i + index) * 4] > thred 
			&& Point[(i + index) * 4] < up_thred)
			NUM = NUM + 1;
		else
			break;
	}
	int NUM2 = 0;
	for (int i = 0; i < index; ++i)
	{
		if (Point[(index - i - 1) * 4] > thred 
			&& Point[(index - i - 1) * 4] < up_thred)
			NUM2 = NUM2 + 1;
		else
			break;
	}
	return NUM + NUM2;
}


__device__ void LineregiongrowingSET_NEW(float *Point0, float *Point1, float *Point2, int Point_num, int Index, double thred, float up_thred,
	                                     int *res_num)
{
	int num0 = Lineregiongrowing(Point0, Index, thred, up_thred, Point_num);
	int num1 = Lineregiongrowing(Point1, Index, thred, up_thred, Point_num);
	int num2 = Lineregiongrowing(Point2, Index, thred, up_thred, Point_num);
	res_num[0] = num0;
	res_num[1] = num1;
	res_num[2] = num2;
}


__device__ void LineregiongrowingSET(float *Point0, float *Point1, float *Point2, int Point_num, int Index, double thred, int *res_num)
{
	int num0 = Lineregiongrowing(Point0, Index, thred, Point_num);
	int num1 = Lineregiongrowing(Point1, Index, thred, Point_num);
	int num2 = Lineregiongrowing(Point2, Index, thred, Point_num);
	res_num[0] = num0;
	res_num[1] = num1;
	res_num[2] = num2;
}


__global__ void conv_mult(float *ori, float *kernel, float *res, int dim_x, int dim_y, int dim_z, int kernel_dim, int total_pixel_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int padding = (kernel_dim - 1) / 2;
	if (idx < total_pixel_num)
	{
		int z = idx / (dim_x * dim_y);
		int xy = idx % (dim_x * dim_y);
		int y = xy / dim_x;
		int x = xy % dim_x;
		float value = 0.0;
		for (int i = 0; i < kernel_dim; ++i)   //z
		{
			for (int j = 0; j < kernel_dim; ++j)  //y
			{
				for (int k = 0; k < kernel_dim; ++k) //x
				{
					/*if (x == 0 && y == 5 && z == 5)
					{
						printf("%f, %f,%d, %d, %d, %d, %d, %d\n", ori[(x + k) + (y + j) * (dim_x + padding * 2) + (z + i) * (dim_x + padding * 2) * (dim_y + padding * 2)],
							kernel[k + j * kernel_dim + i * kernel_dim * kernel_dim], k, j, i, x + k, y + j, z + i);
					}*/
					value += ori[(x + k) + (y + j) * (dim_x + padding * 2) + (z + i) * (dim_x + padding * 2) * (dim_y + padding * 2)]
						* kernel[k + j * kernel_dim + i * kernel_dim * kernel_dim];
				}
			}
		}
		/*if (x == 0 && y == 5 && z == 5)
		{
			printf("%f\n", value);
		}*/
		res[idx] = value;
	}
}
//__global__ void conv_mult_new(unsigned short *ori, float *kernel, float *res, int dim_x, int dim_y, int dim_z, int kernel_dim, int total_pixel_num, int kernel_num)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int padding = (kernel_dim - 1) / 2;
//	if (idx < total_pixel_num)
//	{
//		int z = idx / (dim_x * dim_y);
//		int xy = idx % (dim_x * dim_y);
//		int y = xy / dim_x;
//		int x = xy % dim_x;
//		float value = 0.0;
//		for (int i = 0; i < kernel_dim; ++i)   //z
//		{
//			for (int j = 0; j < kernel_dim; ++j)  //y
//			{
//				for (int k = 0; k < kernel_dim; ++k) //x
//				{
//					/*	value += ori[(x + k) + (y + j) * (dim_x + padding * 2) + (z + i) * (dim_x + padding * 2) * (dim_y + padding * 2)]
//							* kernel[k + j * kernel_dim + i * kernel_dim * kernel_dim];*/  // conv with xyz
//
//					value += ori[(z + i) + (y + j) * (dim_x + padding * 2) + (x + k) * (dim_x + padding * 2) * (dim_y + padding * 2)]
//						   * kernel[k + j * kernel_dim + i * kernel_dim * kernel_dim];
//				}
//			}
//		}
//		res[idx] = value;
//	}
//}


__global__ void conv_mult(unsigned short *ori, float *kernel, float *res, int dim_x, int dim_y, int dim_z, int kernel_dim, int total_pixel_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int padding = (kernel_dim - 1) / 2;
	if (idx < total_pixel_num)
	{
		int z = idx / (dim_x * dim_y);
		int xy = idx % (dim_x * dim_y);
		int y = xy / dim_x;
		int x = xy % dim_x;
		float value = 0.0;
		for (int i = 0; i < kernel_dim; ++i)   //z
		{
			for (int j = 0; j < kernel_dim; ++j)  //y
			{
				for (int k = 0; k < kernel_dim; ++k) //x
				{
					// conv with xyz
					/*value += ori[(x + k) + (y + j) * (dim_x + padding * 2) + (z + i) * (dim_x + padding * 2) * (dim_y + padding * 2)]
						* kernel[k + j * kernel_dim + i * kernel_dim * kernel_dim];  */

					value += ori[(z + i) + (y + j) * (dim_x + padding * 2) + (x + k) * (dim_x + padding * 2) * (dim_y + padding * 2)]
						* kernel[k + j * kernel_dim + i * kernel_dim * kernel_dim];
				}
			}
		}
		/*if (x == 0 && y == 5 && z == 5)
		{
			printf("%f\n", value);
		}*/
		res[idx] = value;
	}
}


__global__ void pad_img(float *ori, float *tar, int padding_x, int padding_y, int padding_z,
	int total_pixel_num, int dim_x, int dim_y, int dim_z)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		int z = idx / (dim_x * dim_y);
		int xy = idx % (dim_x * dim_y);
		int y = xy / dim_x;
		int x = xy % dim_x;
		tar[x + padding_x + (y + padding_y) * (dim_x + padding_x * 2) + (z + padding_z) * (dim_x + padding_x * 2) * (dim_y + padding_y * 2)] = ori[idx];
	}
}


__global__ void obtain_ids(unsigned char *res, float *src, int total_pixel_num, int kernel_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		int res_id = 0;
		float value = -1000.0;
		for (int i = 0; i < kernel_num; ++i)
		{
			float tmp_res = src[i * total_pixel_num + idx];
			if (tmp_res > value)
			{
				value = tmp_res;
				res_id = i;
			}
		}
		res[idx] = res_id;
	}
}


__global__ void pad_img(unsigned short *ori, unsigned short *tar, int padding_x, int padding_y, int padding_z,
	int total_pixel_num, int dim_x, int dim_y, int dim_z)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		int z = idx / (dim_x * dim_y);
		int xy = idx % (dim_x * dim_y);
		int y = xy / dim_x;
		int x = xy % dim_x;
		tar[x + padding_x + (y + padding_y) * (dim_x + padding_x * 2) + (z + padding_z) * (dim_x + padding_x * 2) * (dim_y + padding_y * 2)] = ori[idx];
	}
}


__global__ void
Inttofloat(unsigned short *lrImage_d, float *tar, int total_pixel_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		tar[idx] = lrImage_d[idx];
	}
}


__device__ bool calc_ratio(int *center, unsigned short *lrImage_d, int dim_x, int dim_y, int dim_z)
{
	double dif = -1000.0;
	double I = lrImage_d[center[0] + center[1] * dim_x + center[2] * dim_x * dim_y];
	bool negative = false;
	for (int x = 0; x < 3; ++x)
	{
		for (int y = 0; y < 3; ++y)
		{
			for (int z = 0; z < 3; ++z)
			{
				double temp_dif = I
					- lrImage_d[center[0] + x - 1 + (center[1] + y - 1) * dim_x + (center[2] + z - 1) * dim_x * dim_y];
				if (std::abs(temp_dif) > dif)
				{
					dif = temp_dif;
					if (temp_dif < 0)
						negative = true;
					else
						negative = false;
				}
			}
		}
	}
	return negative;
	
	/*if (double(dif) / I < 1.0 && !negative)
		return (1 - double(dif) / I) * 0.5;
	
	return 1.0;*/
}


__global__ void calc_valid_number(unsigned short *lrImage_d, int total_pixel_num, int *valid)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num && lrImage_d[idx] > 0.0001)
	{
		atomicAdd(valid, 1);
	}
}


__global__ void calc_mean(unsigned short *lrImage_d, int total_pixel_num, double *sum, int nonzero)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		double mean = float(lrImage_d[idx]) / nonzero;
		atomicAdd(sum, mean);
	}
}


__global__ void calc_std(unsigned short *lrImage_d, int total_pixel_num, double mean, double *sum, int nonzero)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		float square = (lrImage_d[idx] - mean) * (lrImage_d[idx] - mean) / nonzero;
		atomicAdd(sum, square);
	}
}


__global__ void calc_std(unsigned short *lrImage_d, int total_pixel_num, float mean, float *sum)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		float square = (lrImage_d[idx] - mean) * (lrImage_d[idx] - mean) / total_pixel_num;
		atomicAdd(sum, square);
	}
}


__global__ void
calc_feature_map(unsigned short *lrImage_d, unsigned char *dirset_d, float *ss_d, int dim_x, int dim_y, int dim_z, 
	unsigned char *predict, int total_pixel_num, int ss_x, int ss_y, int ss_z, int level, int type_id, double thred, double thred2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		int i = idx / (dim_x * dim_y);    // z
		int ii = idx % (dim_x * dim_y);  
		int j = ii / dim_x;              // y
		int k = ii % dim_x;              // x
		/*int id = dirset_d[k + dim_x * j + ii];*/
		int id = dirset_d[idx];
		float dirx[3] = { ss_d[id + 2 * ss_x] - ss_d[id],
						  ss_d[id + 2 * ss_x + 1 * ss_x * ss_y] - ss_d[id + 1 * ss_x * ss_y],
						  ss_d[id + 2 * ss_x + 2 * ss_x * ss_y] - ss_d[id + 2 * ss_x * ss_y] };
		float res[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		Coordinates(dirx, res);
		float *Point_v;
		int center[3] = { k, j, i };
		float Point0[4 * (2 * 4 + 1)], Point1[4 * (2 * 4 + 1)], Point2[4 * (2 * 4 + 1)];
		ImgLineSampleSet(center, res, 4, lrImage_d, dim_x, dim_y, dim_z, Point0, Point1, Point2);
		double ratio = 1.0;
		bool negative = true;
		if(k > 0 && k < dim_x - 1 && j > 0 && j < dim_y - 1 && i > 0 && i < dim_z - 1)
			negative = calc_ratio(center, lrImage_d, dim_x, dim_y, dim_z);
		ratio = 0.2;

		float thres = Point0[4 * 4];

		if (thres != 0 )
		{
			double a = 0.1 * (7 - level) * thres * ratio; // 7 - lve
			//double b = (7 - level) * 0.3;
			double thre_sub = a;
			//if (a > b)
				//thre_sub = a;
			if ((thres - thre_sub - 1) > thred)  // 20  lightsheet 10.0
			{
				int res_num[3] = { 0, 0, 0 };
				/*LineregiongrowingSET(Point0, Point1, Point2, 4 * (2 * 4 + 1), 4, thres - thre_sub - 1, res_num);*/
				if (type_id > 2)
					LineregiongrowingSET_NEW(Point0, Point1, Point2, 4 * (2 * 4 + 1), 4, thres - thre_sub - 1, thres + thre_sub, res_num);
				else
					LineregiongrowingSET_NEW(Point0, Point1, Point2, 4 * (2 * 4 + 1), 4, thres - thre_sub - 1, 70000, res_num);
				double p = 0.0;
				if ((type_id % 3) == 0)
				{
					p = double(res_num[0]) / 9.0;
				}
				else if ((type_id % 3) == 1)
				{
					p = double(res_num[1] * res_num[2]) / (9 * 9);
				}
				else if ((type_id % 3) == 2)
				{
					/*p = double(res_num[0] * res_num[1] * res_num[2]) / (9 * 9 * 9);*/
					double num = res_num[0];
					if (num < res_num[1])
					{
						num = res_num[1];
					}
					if (num < res_num[2])
						num = res_num[2];
					p = double(num) / 9;
				}

				predict[idx] = int((1 - p) * 255);
			}
			else
			{
				predict[idx] = 0.0;
			}
		}
		/*else
		{
			if (level < 3)
			{
				predict[idx] = 0.0;
			}
			else
			{
				double p = double(1 * 1) / (9 * 9);
				predict[idx] = int((1 - p) * 255);
			}
		}*/
	}
}

__global__ void
calc_feature_map_with_meanstd(unsigned short *lrImage_d, unsigned char *dirset_d, float *ss_d, int dim_x, int dim_y, int dim_z,
	unsigned char *predict, int total_pixel_num, int ss_x, int ss_y, int ss_z, int level, int type_id, double thred, double thred2,
	double mean, double std)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_pixel_num)
	{
		int i = idx / (dim_x * dim_y);    // z
		int ii = idx % (dim_x * dim_y);
		int j = ii / dim_x;              // y
		int k = ii % dim_x;              // x
		/*int id = dirset_d[k + dim_x * j + ii];*/
		int id = dirset_d[idx];
		float dirx[3] = { ss_d[id + 2 * ss_x] - ss_d[id],
						  ss_d[id + 2 * ss_x + 1 * ss_x * ss_y] - ss_d[id + 1 * ss_x * ss_y],
						  ss_d[id + 2 * ss_x + 2 * ss_x * ss_y] - ss_d[id + 2 * ss_x * ss_y] };
		float res[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		Coordinates(dirx, res);
		float *Point_v;
		int center[3] = { k, j, i };
		float Point0[4 * (2 * 4 + 1)], Point1[4 * (2 * 4 + 1)], Point2[4 * (2 * 4 + 1)];
		ImgLineSampleSet(center, res, 4, lrImage_d, dim_x, dim_y, dim_z, Point0, Point1, Point2);
		double ratio = 1.0;
		bool negative = true;
		/*if (k > 0 && k < dim_x - 1 && j > 0 && j < dim_y - 1 && i > 0 && i < dim_z - 1)
			negative = calc_ratio(center, lrImage_d, dim_x, dim_y, dim_z);*/
		
		float thres = Point0[4 * 4];
		//if (thres > mean  + 1.1 * std -  std * 0.15 * (level + 1))
	   if (1/*thres > mean + 1.1 * std - std * 0.15 * (level + 1)*/)
		{
			//double a = 0.1 * (7 - level * 0.5) * std; // 7 - lve
			double a = thred2 * (1 + level) * std; // 7 - lve // 0.4, 0.2, 0.1
			/*if (thres < mean + std)
				a = 0.2 * (1 + level) * std;*/
			//a = 0.1 * (7 - level) * thres;
			//double b = (7 - level) * 0.3;
			double thre_sub = a;
			//if (a > b)
				//thre_sub = a;
			if (1/*(thres - thre_sub - 1) > thred*/)  // 20  lightsheet 10.0
			{
				int res_num[3] = { 0, 0, 0 };
				/*LineregiongrowingSET(Point0, Point1, Point2, 4 * (2 * 4 + 1), 4, thres - thre_sub - 1, res_num);*/
				if (type_id > 2)
					LineregiongrowingSET_NEW(Point0, Point1, Point2, 4 * (2 * 4 + 1), 4, thres - thre_sub - 1, thres + thre_sub, res_num);
				else
					LineregiongrowingSET_NEW(Point0, Point1, Point2, 4 * (2 * 4 + 1), 4, thres - thre_sub - 1, 70000, res_num);
				double p = 0.0;
				if ((type_id % 3) == 0)
				{
					p = double(res_num[0]) / 9.0;
				}
				else if ((type_id % 3) == 1)
				{
					p = double(res_num[1] * res_num[2]) / (9 * 9);
				}
				else if ((type_id % 3) == 2)
				{
					/*p = double(res_num[0] * res_num[1] * res_num[2]) / (9 * 9 * 9);*/
					double num = res_num[0];
					if (num < res_num[1])
					{
						num = res_num[1];
					}
					if (num < res_num[2])
						num = res_num[2];
					p = double(num) / 9;
				}

				predict[idx] = int((1 - p) * 255);
			}
			else
			{
				predict[idx] = 0.0;
			}
		}
		/*else
		{
			if (level < 3)
			{
				predict[idx] = 0.0;
			}
			else
			{
				double p = double(1 * 1) / (9 * 9);
				predict[idx] = int((1 - p) * 255);
			}
		}*/
	}
}

void compute_stride(const int * size, int*stride)
{
	for (int i = 4; i >= 0; i--)
		stride[i] = (i == 4) ? 1 : size[i + 1] * stride[i + 1];
}


extern "C" void conv_new(Voxel_L &lrImage, std::vector<Eigen::Tensor<float, 3> > kernels, int dim_x, int dim_y, int dim_z, int kernel_dim, bool reverse_xyz, int padding,
	std::vector<Eigen::Tensor<float, 3> > &res_C, Eigen::Tensor<unsigned char, 3> &id_res_c)
{
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	int total_pixel_num = lrImage.dim_x * lrImage.dim_y * lrImage.dim_z;
	unsigned short* lrImage_d;
	cudaMalloc(&lrImage_d, sizeof(unsigned short) * lrImage.dim_x * lrImage.dim_y * lrImage.dim_z);
	for (int i = 0; i < lrImage.dim_z; ++i)
	{
		cudaMemcpy(lrImage_d + i * lrImage.dim_x * lrImage.dim_y, lrImage.data_ptr[i], sizeof(unsigned short) *
			lrImage.dim_x * lrImage.dim_y, cudaMemcpyHostToDevice);
	}
	float *f_lrImage_d;
	cudaMalloc(&f_lrImage_d, sizeof(float) * lrImage.dim_x * lrImage.dim_y * lrImage.dim_z);
	Inttofloat << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (lrImage_d, f_lrImage_d, total_pixel_num);

	float *d_output;
	
	// 0 创建cudnn句柄
	cudnnHandle_t cudnn;
	auto cudnnHandle = cudnnCreate(&cudnn);
	if (cudnnHandle != CUDNN_STATUS_SUCCESS) {
		std::cout << "创建cudnn句柄：失败！" << std::endl;

	}

	//1 创建数据和计算相关描述符
	// 1.1 创建输入张量描述符
	int q = 1, r = 1, m = dim_x, n = dim_y, p = dim_z;
	//这里创建5维矩阵的原因是高维度卷积计算时，官方文档推荐使用 >= 4维的张量进行计算，不需要的维度定义为1即可
	int inputDims[5] = { q,r,m, n, p }; // 输入张量的尺寸 
	int input_stride[5]; //输入张量描述符的步长------**重要**重要**重要**
	compute_stride(inputDims, input_stride);
	cudnnTensorDescriptor_t inputDesc;//输入张量描述符
	cudnnCreateTensorDescriptor(&inputDesc);
	cudnnStatus_t status = cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 5, inputDims, input_stride);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cout << "创建输入张量描述符：失败！" << std::endl;

	}

	int kernel_num = 108;
	// 1.2 创建卷积核描述符
	int filterDims[5] = { kernel_num, 1, 13, 13, 13 };   // 卷积核尺寸
	cudnnFilterDescriptor_t filterDesc;
	cudnnCreateFilterDescriptor(&filterDesc);
	status = cudnnSetFilterNdDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filterDims);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cout << "创建卷积核描述符：失败！" << std::endl;

	}

	// 1.3 创建卷积运算操作描述符
	int  conmv_padA[3] = { 6, 6, 6 };//填充，表示沿各个维度补0的数量，是为了解决卷积后数据尺寸缩小的问题，设为全0则表示不需要填充
	int conv_filterStrideA[3] = { 1,1,1 };//卷积时使用卷积核的步长，全1表示不跳过，均为单步步进
	int conv_dilationA[3] = { 1,1,1 };//arrayLength（arg 2)数组所指示的每个维度膨胀因子,这个参数对是对卷积核操作的，某个维度膨胀系数>1时，会把卷积沿这个维度放大，中间的缺失数据用0补齐；全1表示无膨胀
	cudnnConvolutionDescriptor_t convDesc;
	cudnnCreateConvolutionDescriptor(&convDesc);
	status = cudnnSetConvolutionNdDescriptor(convDesc, 3, conmv_padA, conv_filterStrideA, conv_dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cout << "创建卷积操作描述符：失败！" << std::endl;

	}

	// 1.4 计算输出张量的尺寸,自己手算也可以，但是还是建议用它的函数来计算，刚好可以验证之前的描述符创建的是否满足自己预期
	// outputDim = 1 + ( inputDim + 2.*pad - (((filterDim-1).*dilation)+1) )./convolutionStride
	int outputDims[5];
	status = cudnnGetConvolutionNdForwardOutputDim(convDesc, inputDesc, filterDesc, 5, outputDims);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cout << "计算输出张量的尺寸：失败！" << std::endl;

	}
	
	/*std::cout << "Output size: ";
	for (int i = 0; i < sizeof(outputDims) / sizeof(float); i++) {
		if (i < sizeof(outputDims) / sizeof(float) - 1) {
			std::cout << outputDims[i];
			std::cout << " X ";
		}
	}
	std::cout << outputDims[sizeof(outputDims) / sizeof(float) - 1] << std::endl;*/

	// 1.5 创建输出张量描述符
	cudnnTensorDescriptor_t outputDesc;
	cudnnCreateTensorDescriptor(&outputDesc);
	int output_stride[5]; //输出张量描述符的步长
	compute_stride(outputDims, output_stride);
	status = cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 5, outputDims, output_stride);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cout << "创建输出张量描述符：失败！" << std::endl;

	}

	// 2 数据和计算内存空间分配与初始化
	//2.1 计算各变量所需内存大小
	size_t in_bytes = 0;//输入张量所需内存
	status = cudnnGetTensorSizeInBytes(inputDesc, &in_bytes);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "fail to get bytes of in tensor: " << cudnnGetErrorString(status) << std::endl;

	}
	size_t  out_bytes = 0;//输出张量所需内存
	status = cudnnGetTensorSizeInBytes(outputDesc, &out_bytes);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "fail to get bytes of out tensor: " << cudnnGetErrorString(status) << std::endl;

	}
	size_t filt_bytes = 1;//卷积核所需内存
	for (int i = 0; i < sizeof(filterDims) / sizeof(int); i++) {
		filt_bytes *= filterDims[i];
	}
	filt_bytes *= sizeof(float);
	//自动寻找最优卷积计算方法,函数自动选择的最优算法保存在perfResults结构体中
	int returnedAlgoCount = 0;
	cudnnConvolutionFwdAlgoPerf_t perfResults;
	status = cudnnGetConvolutionForwardAlgorithm_v7(cudnn, inputDesc, filterDesc, convDesc, outputDesc, 1, &returnedAlgoCount, &perfResults);
	if (returnedAlgoCount != 1 || status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "自动适配卷积计算方法：失败！" << std::endl;

	}
	//计算卷积操作所需的内存空间大小，保存在workspace_bytes变量中
	size_t workspace_bytes{ 0 };
	status = cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, perfResults.algo, &workspace_bytes);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cout << "计算卷积操作所需的内存空间：失败！" << std::endl;

	}

	//2.2 判断GPU上是否有足够的内存空间用于计算
	size_t request = in_bytes + out_bytes + workspace_bytes + filt_bytes;
	size_t cudaMem_free = 0, cudaMem_total = 0;
	cudaError_t cuda_err = cudaMemGetInfo(&cudaMem_free, &cudaMem_total);
	if (cuda_err != cudaSuccess) {
		std::cerr << "fail to get mem info: " << cudaGetErrorString(cuda_err) << std::endl;

	}
	if (request > cudaMem_free) {
		std::cout << in_bytes / (1024.0 * 1024.0 * 1024.0) << " " << out_bytes / (1024.0 * 1024.0 * 1024.0) << " "
			<< workspace_bytes / (1024.0 * 1024.0 * 1024.0) << " " << filt_bytes / (1024.0 * 1024.0 * 1024.0) << std::endl;
		std::cerr << request / (1024.0 * 1024.0 * 1024.0) << " " << cudaMem_free / (1024.0 * 1024.0 * 1024.0) << " not enough gpu memory to run" << std::endl;

	}

	//2.2 在主机上分配内存存储输入张量、卷积核和输出张量


	// 2.3 数据初始化
	// cuda要求输入是一维数据（NCHW格式）,所以将原数据reshape成一行数据,输入到GPU之后会根据描述符中的维度参数还原的，所以不用担心,
	// 无论多少维的数据， 按顺序reshape为一行即可(但是NCHW格式和NHWC格式的reshape方式是不一样的，一定要注意，上下文匹配即可）
	// 2.3.1 输入张量   

	//2.3.2 卷积核数据，我随便举的例子，可以根据自己的需求自己改，只要维度和前面定义的卷积核维度一致即可


	// 2.4 在设备（gpu)上分配内存空间
	cudaMalloc(&d_output, out_bytes);
	void* d_workspace{ nullptr };
	cudaMalloc(&d_workspace, workspace_bytes);

	// 3 将输入张量和卷积核拷贝到设备

	float* kernel_d;
	cudaMalloc(&kernel_d, sizeof(float) * kernel_dim * kernel_dim * kernel_dim * kernel_num);
	for (int i = 0; i < kernel_num; ++i)
		cudaMemcpy(kernel_d + i * kernel_dim * kernel_dim * kernel_dim, kernels[i].data(), sizeof(float) * kernel_dim * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);


	// 4 进行卷积计算
	float alpha = 1.0f, beta = 0.0f;
	status = cudnnConvolutionForward(cudnn, &alpha, inputDesc, f_lrImage_d, filterDesc, kernel_d, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, d_workspace, workspace_bytes, &beta, outputDesc, d_output);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cout << "卷积计算过程：失败！" << std::endl;

	}
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	// 5 将输出矩阵拷贝回主机
	//Eigen::Tensor<float, 3> res_c(128, 128, 128);
	/*for (int i = 0; i < 108; ++i)
	{
		cudaMemcpy(res_c.data(), d_output + i * 128 * 128 * 128, sizeof(float) * 128 * 128 * 128, cudaMemcpyDeviceToHost);
		res_C.push_back(res_c);
	}*/
	//打印输出矩阵
	/*for (int i = 0; i < out_bytes; i++) {
		std::cout << output[i] << " ";
	}*/


	// 6 释放资源
	//6.1 释放三个变量占用的内存

	
	//6.2 释放描述符占用的内存
	status = cudnnDestroyTensorDescriptor(inputDesc);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "fail to destroy input tensor desc: " << cudnnGetErrorString(status) << std::endl;

	}
	status = cudnnDestroyFilterDescriptor(filterDesc);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "fail to destroy filter tensor desc: " << cudnnGetErrorString(status) << std::endl;

	}
	status = cudnnDestroyConvolutionDescriptor(convDesc);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "fail to destroy conv desc: " << cudnnGetErrorString(status) << std::endl;

	}
	status = cudnnDestroyTensorDescriptor(outputDesc);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "fail to destroy outout tensor desc: " << cudnnGetErrorString(status) << std::endl;

	}
	//6.3 释放cudnn句柄内存
	status = cudnnDestroy(cudnn);
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "fail to destroy cudnn handle" << std::endl;

	}
	//6.4 释放主机上存储数组的内存
	std::cout << "Done." << std::endl;

	float time;
	cudaEventElapsedTime(&time, start, end);
	std::cout << "GPU Time: " << time << std::endl;
	
	
	/*for (int i = 0; i < kernels.size(); ++i)
	{
		float* kernel_d;
		cudaMalloc(&kernel_d, sizeof(float) * kernel_dim * kernel_dim * kernel_dim);
		cudaMemcpy(kernel_d, kernels[i].data(), sizeof(float) * kernel_dim * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);

		Eigen::Tensor<float, 3> res_c(128, 128, 128);
		conv_mult << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (padded_img_d, kernel_d, res + i * total_pixel_num,
			lrImage.dim_x, lrImage.dim_y, lrImage.dim_z, kernel_dim, total_pixel_num);
		cudaMemcpy(res_c.data(), res + i * total_pixel_num, sizeof(float) * 128 * 128 * 128, cudaMemcpyDeviceToHost);
		res_C.push_back(res_c);
	}*/

	unsigned char *id_res;
	cudaMalloc(&id_res, sizeof(unsigned char*) * total_pixel_num);
	obtain_ids << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (id_res, d_output, total_pixel_num, kernels.size());
	cudaMemcpy(id_res_c.data(), id_res, sizeof(unsigned char) * total_pixel_num, cudaMemcpyDeviceToHost);

	// free memory
	{
		cudaFree(lrImage_d);
		cudaFree(f_lrImage_d);
		cudaFree(d_output);
		cudaFree(d_workspace);
		cudaFree(kernel_d);
		cudaFree(id_res); 
		cuda_err = cudaFree(d_output);
		
	}
}


extern "C" void conv(Voxel_L &lrImage, std::vector<Eigen::Tensor<float, 3> > kernels, int dim_x, int dim_y, int dim_z, int kernel_dim, bool reverse_xyz, int padding,
	std::vector<Eigen::Tensor<float, 3> > &res_C, Eigen::Tensor<unsigned char, 3> &id_res_c)
{	
	unsigned short* lrImage_d;
	cudaMalloc(&lrImage_d, sizeof(unsigned short) * lrImage.dim_x * lrImage.dim_y * lrImage.dim_z);
	for (int i = 0; i < lrImage.dim_z; ++i)
	{
		cudaMemcpy(lrImage_d + i * lrImage.dim_x * lrImage.dim_y, lrImage.data_ptr[i], sizeof(unsigned short) *
			lrImage.dim_x * lrImage.dim_y, cudaMemcpyHostToDevice);
	}
	unsigned short* padded_img_d;
	cudaMalloc(&padded_img_d, sizeof(unsigned short) * (lrImage.dim_x + padding * 2)
											* (lrImage.dim_y + padding * 2)
											* (lrImage.dim_z + padding * 2));
	int total_padded_num = (lrImage.dim_x + padding * 2) * (lrImage.dim_y + padding * 2) * (lrImage.dim_z + padding * 2);
	unsigned short initial_val = 0;
	initialize <<< (total_padded_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (padded_img_d, initial_val, total_padded_num);

	int total_pixel_num = lrImage.dim_x * lrImage.dim_y * lrImage.dim_z;
	pad_img << <(total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (lrImage_d, 
		padded_img_d, padding, padding, padding, total_pixel_num, lrImage.dim_x, lrImage.dim_y, lrImage.dim_z);

	float *res;
	cudaMalloc(&res, sizeof(float) * lrImage.dim_x * lrImage.dim_y * lrImage.dim_z * kernels.size());

	// original version.
	//for (int i = 0; i < kernels.size(); ++i)
	//{
	//	float* kernel_d;
	//	cudaMalloc(&kernel_d, sizeof(float) * kernel_dim * kernel_dim * kernel_dim);
	//	cudaMemcpy(kernel_d, kernels[i].data(), sizeof(float) * kernel_dim * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);

	//	Eigen::Tensor<float, 3> res_c(128, 128, 128);
	//	conv_mult << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >>> (padded_img_d, kernel_d, res + i * total_pixel_num,
	//		lrImage.dim_x, lrImage.dim_y, lrImage.dim_z, kernel_dim, total_pixel_num);
	//	/*cudaMemcpy(res_c.data(), res + i * total_pixel_num, sizeof(float) * 128 * 128 * 128, cudaMemcpyDeviceToHost);
	//	res_C.push_back(res_c);*/
	//}
	
	for (int i = 0; i < kernels.size(); ++i)
	{
		float* kernel_d;
		cudaMalloc(&kernel_d, sizeof(float) * kernel_dim * kernel_dim * kernel_dim);
		cudaMemcpy(kernel_d, kernels[i].data(), sizeof(float) * kernel_dim * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);

		Eigen::Tensor<float, 3> res_c(128, 128, 128);
		conv_mult << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (padded_img_d, kernel_d, res + i * total_pixel_num,
			lrImage.dim_x, lrImage.dim_y, lrImage.dim_z, kernel_dim, total_pixel_num);
		/*cudaMemcpy(res_c.data(), res + i * total_pixel_num, sizeof(float) * 128 * 128 * 128, cudaMemcpyDeviceToHost);
		res_C.push_back(res_c);*/
	}

	unsigned char *id_res;
	cudaMalloc(&id_res, sizeof(unsigned char*) * total_pixel_num);
	obtain_ids << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (id_res, res, total_pixel_num, kernels.size() );
	cudaMemcpy(id_res_c.data(), id_res, sizeof(unsigned char) * total_pixel_num, cudaMemcpyDeviceToHost);

	// free memory
	{
		cudaFree(lrImage_d);
		cudaFree(padded_img_d);
		cudaFree(res);
		cudaFree(id_res);
	}
}


__global__ void initialize(float *src, float val, int num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
	{
		src[idx] = val;
	}
}


__global__ void initialize(unsigned short *src, unsigned short val, int num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
	{
		src[idx] = val;
	}
}

//__global__ void initialize(unsigned short *src, unsigned short val, int num)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < num)
//	{
//		src[idx] = val;
//	}
//}


extern "C" void test_conv(std::vector<Eigen::Tensor<float, 3>> &oris, Eigen::Tensor<float, 3> &kernel, std::vector<Eigen::Tensor<float, 3>> &output)
{
	int padding = 1;
	int kernel_dim = 3;
	for (int ori_id = 0; ori_id < oris.size(); ori_id++)
	{
		float* lrImage_d;
		cudaMalloc(&lrImage_d, sizeof(float) * oris[ori_id].dimension(0) * oris[ori_id].dimension(1) * oris[ori_id].dimension(2));
		cudaMemcpy(lrImage_d, oris[ori_id].data(), sizeof(float) * oris[ori_id].dimension(0) * oris[ori_id].dimension(1) * oris[ori_id].dimension(2),
			cudaMemcpyHostToDevice);
		
		float* padded_img_d;
		cudaMalloc(&padded_img_d, sizeof(float) * (oris[ori_id].dimension(0) + padding * 2) * (oris[ori_id].dimension(1) + padding * 2)
			* (oris[ori_id].dimension(2) + padding * 2));
	/*	cudaMemset(padded_img_d, 1.0, sizeof(float) * (oris[ori_id].dimension(0) + padding * 2) * (oris[ori_id].dimension(1) + padding * 2)
			* (oris[ori_id].dimension(2) + padding * 2));*/
		initialize << < ((oris[ori_id].dimension(0) + padding * 2) * (oris[ori_id].dimension(1) + padding * 2)
			* (oris[ori_id].dimension(2) + padding * 2) + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (padded_img_d, 0.0f,
			(oris[ori_id].dimension(0) + padding * 2) * (oris[ori_id].dimension(1) + padding * 2)
				* (oris[ori_id].dimension(2) + padding * 2));

		int total_pixel_num = oris[ori_id].dimension(0) * oris[ori_id].dimension(1) * oris[ori_id].dimension(2);
		pad_img << <(total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (lrImage_d,
			padded_img_d, padding, padding, padding, total_pixel_num, oris[ori_id].dimension(0), oris[ori_id].dimension(1), oris[ori_id].dimension(2));
		Eigen::Tensor<float, 3> padded_img(oris[ori_id].dimension(0) + 2, oris[ori_id].dimension(1) + 2, oris[ori_id].dimension(2) + 2);
		cudaMemcpy(padded_img.data(), padded_img_d, sizeof(float) * (oris[ori_id].dimension(0) + padding * 2) * (oris[ori_id].dimension(1) + padding * 2) * (oris[ori_id].dimension(2) + padding * 2), cudaMemcpyDeviceToHost);
		
		float* kernel_d;
		cudaMalloc(&kernel_d, sizeof(float) * kernel_dim * kernel_dim * kernel_dim);
		cudaMemcpy(kernel_d, kernel.data(), sizeof(float) * kernel_dim * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);

		float *res;
		cudaMalloc(&res, sizeof(float) * oris[ori_id].dimension(0) * oris[ori_id].dimension(1) * oris[ori_id].dimension(2));
		conv_mult << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (padded_img_d, kernel_d, res, oris[ori_id].dimension(0),
			oris[ori_id].dimension(1), oris[ori_id].dimension(2), 3, total_pixel_num);

		Eigen::Tensor<float, 3> tmp_res(oris[ori_id].dimension(0), oris[ori_id].dimension(1), oris[ori_id].dimension(2));
		cudaMemcpy(tmp_res.data(), res, sizeof(float) * oris[ori_id].dimension(0) * oris[ori_id].dimension(1) * oris[ori_id].dimension(2), cudaMemcpyDeviceToHost);
		output.push_back(tmp_res);
	}
}


extern "C" void calc_direction_id(std::vector<Eigen::Tensor<float, 3> > kernels, Voxel_L &lrImage, std::vector<Eigen::Tensor<float, 3> > &res,
	Eigen::Tensor<unsigned char, 3> &idx_res)
{
	conv_new(lrImage, kernels, lrImage.dim_x, lrImage.dim_y, lrImage.dim_z, kernels[0].dimension(0), true, int((kernels[0].dimension(0) - 1) / 2), res,
		idx_res);
}


void calc_mean_std(unsigned short *lrImage_d, int total_pixel_num, double mean_res, double std_res)
{
	
}



extern "C" void fast_iter2(int level, Voxel_L &lrImage, Eigen::Tensor<unsigned char, 3> &dirSet, Eigen::Tensor<float, 3> &ss,
	int ss_x, int ss_y, int ss_z, Voxel_E &predict, int type_id, double thred, double thred2)
{
	unsigned short* lrImage_d;
	cudaMalloc(&lrImage_d, sizeof(unsigned short) * lrImage.dim_x * lrImage.dim_y * lrImage.dim_z);
	for (int i = 0; i < lrImage.dim_z; ++i)
	{
		cudaMemcpy(lrImage_d + i * lrImage.dim_x * lrImage.dim_y, lrImage.data_ptr[i], sizeof(unsigned short) *
			lrImage.dim_x * lrImage.dim_y, cudaMemcpyHostToDevice);
	}

	unsigned char* dirset_d;
	cudaMalloc(&dirset_d, sizeof(unsigned char) * lrImage.dim_x * lrImage.dim_y * lrImage.dim_z);
	cudaMemcpy(dirset_d, dirSet.data(), sizeof(unsigned char) * lrImage.dim_x * lrImage.dim_y * lrImage.dim_z, cudaMemcpyHostToDevice);
	
	float *ss_d;
	cudaMalloc(&ss_d, sizeof(float) * ss_x * ss_y * ss_z);
	cudaMemcpy(ss_d, ss.data(), sizeof(float) * ss_x * ss_y * ss_z, cudaMemcpyHostToDevice);

	const int d = lrImage.dim_z, w = lrImage.dim_y, h = lrImage.dim_x;
	unsigned char *Predict;
	cudaMalloc(&Predict, sizeof(unsigned char) * w * h * d);
	int total_pixel_num = d * w * h;

	double *mean_d, mean = 0.0;
	int *valid_num, valid = 0;
	cudaMalloc(&valid_num, sizeof(int));
	cudaMemcpy(valid_num, &valid, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&mean_d, sizeof(double));
	cudaMemcpy(mean_d, &mean, sizeof(double), cudaMemcpyHostToDevice);
	double std = 0.0;
	double *std_d;
	cudaMalloc(&std_d, sizeof(double));
	cudaMemcpy(std_d, &std, sizeof(double), cudaMemcpyHostToDevice);
	
	calc_valid_number << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (lrImage_d, total_pixel_num, valid_num);
	cudaMemcpy(&valid, valid_num, sizeof(int), cudaMemcpyDeviceToHost);

	calc_mean << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (lrImage_d, total_pixel_num, mean_d, valid);
	cudaMemcpy(&mean, mean_d, sizeof(double), cudaMemcpyDeviceToHost);
	calc_std << < (total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (lrImage_d, total_pixel_num, mean, std_d, valid);
	cudaMemcpy(&std, std_d, sizeof(double), cudaMemcpyDeviceToHost);
	//std = 40.0 * 40.0;
	if(level == 0)
		printf("Valid: %d, Mean : %f, Std : %f\n", valid, mean, std::sqrt(std));

	calc_feature_map_with_meanstd << <(total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >> > (lrImage_d, dirset_d, ss_d, h, w, d, Predict,
		total_pixel_num, ss_x, ss_y, ss_z, level, type_id, thred, thred2, mean, std::sqrt(std) );

	for (int i = 0; i < predict.dim_z; ++i)
	{
		cudaMemcpy(predict.data_ptr[i], Predict + i * predict.dim_x * predict.dim_y, sizeof(unsigned char) * predict.dim_x * predict.dim_y, cudaMemcpyDeviceToHost);
	}
	cudaFree(lrImage_d);
	cudaFree(dirset_d);
	cudaFree(ss_d);
	cudaFree(Predict);
	cudaFree(valid_num);
	cudaFree(mean_d);
	cudaFree(std_d);
}


extern "C" void fast_iter(int level, Voxel_L &lrImage, Voxel_E &dirSet, Eigen::Tensor<float, 3> &ss,
	int ss_x, int ss_y, int ss_z, Voxel_E &predict, int type_id)
{
	unsigned short* lrImage_d;
	cudaMalloc(&lrImage_d, sizeof(unsigned short) * lrImage.dim_x * lrImage.dim_y * lrImage.dim_z);
	for (int i = 0; i < lrImage.dim_z; ++i)
	{
		cudaMemcpy(lrImage_d + i * lrImage.dim_x * lrImage.dim_y, lrImage.data_ptr[i], sizeof(unsigned short) *
			lrImage.dim_x * lrImage.dim_y, cudaMemcpyHostToDevice);
	}

	unsigned char* dirset_d;
	cudaMalloc(&dirset_d, sizeof(unsigned char) * dirSet.dim_x * dirSet.dim_y * dirSet.dim_z);
	for (int i = 0; i < dirSet.dim_z; ++i)
	{
		cudaMemcpy(dirset_d + i * dirSet.dim_x * dirSet.dim_y, dirSet.data_ptr[i], sizeof(unsigned char) * dirSet.dim_x * dirSet.dim_y, cudaMemcpyHostToDevice);
	}

	float *ss_d;
	cudaMalloc(&ss_d, sizeof(float) * ss_x * ss_y * ss_z);
	cudaMemcpy(ss_d, ss.data(), sizeof(float) * ss_x * ss_y * ss_z, cudaMemcpyHostToDevice);

	const int d = lrImage.dim_z, w = lrImage.dim_y, h = lrImage.dim_x;
	unsigned char *Predict;
	cudaMalloc(&Predict, sizeof(unsigned char) * w * h * d);
	int total_pixel_num = d * w * h;
	calc_feature_map<<<(total_pixel_num + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(lrImage_d, dirset_d, ss_d, h, w, d, Predict,
		total_pixel_num, ss_x, ss_y, ss_z, level, type_id, 20.0, 1.0);
	

	for (int i = 0; i < predict.dim_z; ++i)
	{
		cudaMemcpy(predict.data_ptr[i], Predict + i * predict.dim_x * predict.dim_y, sizeof(unsigned char) * predict.dim_x * predict.dim_y, cudaMemcpyDeviceToHost);
	}
	cudaFree(lrImage_d);
	cudaFree(dirset_d);
	cudaFree(ss_d);
	cudaFree(Predict);
}