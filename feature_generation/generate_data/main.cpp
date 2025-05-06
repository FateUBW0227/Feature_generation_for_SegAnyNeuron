#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "voxel.h"
#include <chrono>
#include <iomanip>
#include <tiffio.h>
#include <algorithm>
#include <CL_IO/cl_io.h>
#include "omp.h"
#include "utils.h"
#include <algorithm>
#include <filesystem>
#include <random>

extern "C" void fast_iter(int level, Voxel_L &lrImage, Voxel_E &dirSet, Eigen::Tensor<float, 3> &ss,
	int ss_x, int ss_y, int ss_z, Voxel_E &predict, int type_id);

extern "C" void fast_iter2(int level, Voxel_L &lrImage, Eigen::Tensor<unsigned char, 3> &dirSet, Eigen::Tensor<float, 3> &ss,
	int ss_x, int ss_y, int ss_z, Voxel_E &predict, int type_id, double thred, double thred2);

extern "C" void calc_direction_id(std::vector<Eigen::Tensor<float, 3> > kernels, Voxel_L &lrImage, std::vector<Eigen::Tensor<float, 3> > &res,
	Eigen::Tensor<unsigned char, 3> &idx_res);

extern "C" void test_conv(std::vector<Eigen::Tensor<float, 3>> &oris, Eigen::Tensor<float, 3> &kernel, std::vector<Eigen::Tensor<float, 3>> &res);


double PI = 3.1415926;

bool load_tif(std::string file_name, Voxel_E &voxel) {
	TIFF *in = TIFFOpen(file_name.c_str(), "r");
	if (!in) {
		return false;
	}

	uint16 bitspersample = 8;
	uint16 photo = 1; // 1 : 0 is dark, 255 is white
	uint16 samplesperpixel = 1; // 8bit denote a pixel
	TIFFGetField(in, TIFFTAG_BITSPERSAMPLE, &bitspersample);
	TIFFGetField(in, TIFFTAG_PHOTOMETRIC, &photo); // PHOTOMETRIC_MINISWHITE=0, PHOTOMETRIC_MINISBLACK=1
	TIFFGetField(in, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel);

	int _x, _y, _z;
	TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &_x);
	TIFFGetField(in, TIFFTAG_IMAGELENGTH, &_y);
	_z = TIFFNumberOfDirectories(in);
	voxel.set_shape(_x, _y, _z);

	if (bitspersample == 8) {
		TIFFSetDirectory(in, 0);
		uint8 *data = new uint8[_x];
		for (int nCur = 0; nCur < _z; ++nCur) {
			for (int j = 0; j < _y; ++j) {
				TIFFReadScanline(in, data, j, 0);
				for (int i = 0; i < _x; ++i) {
					voxel(i, j, nCur) = data[i];
				}
			}
			TIFFReadDirectory(in); // read next page
		}
		delete[] data;
	}
	else {
		uint16 *data = new uint16[_x * _y];
		int width(0), height(0);
		TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &width);
		TIFFGetField(in, TIFFTAG_IMAGELENGTH, &height);
		memset(data, 0, sizeof(short) * width * height);
		TIFFSetDirectory(in, 0);
		int StripSize = TIFFStripSize(in);
		int NumberofStrips = TIFFNumberOfStrips(in);
		for (int nCur = 0; nCur < _z; ++nCur) {
			for (int i = 0; i < NumberofStrips; ++i) {
				TIFFReadEncodedStrip(in, i, (data + i * StripSize / sizeof(short)), StripSize); //
			}
			for (int i = 0; i < _x; ++i) {
				for (int j = 0; j < _y; ++j) {
					voxel(i, j, nCur) = data[i + j * _x];
				}
			}
			TIFFReadDirectory(in); // read next page
		}
		delete[] data;
	}
	TIFFClose(in);
	return true;
}


//bool load_tif(std::string file_name, Voxel_L &voxel) {
//	TIFF *in = TIFFOpen(file_name.c_str(), "r");
//	if (!in) {
//		return false;
//	}
//
//	uint16 bitspersample = 16;
//	uint16 photo = 1; // 1 : 0 is dark, 255 is white
//	uint16 samplesperpixel = 1; // 8bit denote a pixel
//	TIFFGetField(in, TIFFTAG_BITSPERSAMPLE, &bitspersample);
//	TIFFGetField(in, TIFFTAG_PHOTOMETRIC, &photo); // PHOTOMETRIC_MINISWHITE=0, PHOTOMETRIC_MINISBLACK=1
//	TIFFGetField(in, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel);
//
//	int _x, _y, _z;
//	TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &_x);
//	TIFFGetField(in, TIFFTAG_IMAGELENGTH, &_y);
//	_z = TIFFNumberOfDirectories(in);
//	voxel.set_shape(_x, _y, _z);
//
//	if (bitspersample == 8) {
//		TIFFSetDirectory(in, 0);
//		uint8 *data = new uint8[_x];
//		for (int nCur = 0; nCur < _z; ++nCur) {
//			for (int j = 0; j < _y; ++j) {
//				TIFFReadScanline(in, data, j, 0);
//				for (int i = 0; i < _x; ++i) {
//					voxel(i, j, nCur) = data[i];
//				}
//			}
//			TIFFReadDirectory(in); // read next page
//		}
//		delete[] data;
//	}
//	else {
//		uint16 *data = new uint16[_x * _y];
//		int width(0), height(0);
//		TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &width);
//		TIFFGetField(in, TIFFTAG_IMAGELENGTH, &height);
//		memset(data, 0, sizeof(short) * width * height);
//		TIFFSetDirectory(in, 0);
//		int StripSize = TIFFStripSize(in);
//		int NumberofStrips = TIFFNumberOfStrips(in);
//		for (int nCur = 0; nCur < _z; ++nCur) {
//			for (int i = 0; i < NumberofStrips; ++i) {
//				TIFFReadEncodedStrip(in, i, (data + i * StripSize / sizeof(short)), StripSize); //
//			}
//			for (int i = 0; i < _x; ++i) {
//				for (int j = 0; j < _y; ++j) {
//					voxel(i, j, nCur) = data[i + j * _x];
//				}
//			}
//			TIFFReadDirectory(in); // read next page
//		}
//		delete[] data;
//	}
//	TIFFClose(in);
//	return true;
//}


void elmentvise_add_direct(Eigen::Tensor<float, 3> &pt, float val)
{
	for (int i = 0; i < pt.dimension(0); ++i)
	{
		for (int j = 0; j < pt.dimension(1); ++j)
		{
			for (int k = 0; k < pt.dimension(2); ++k)
			{
				pt(i, j, k) += val;
			}
		}
	}
}


Eigen::Tensor<float, 3> elmentvise_add(Eigen::Tensor<float, 3> &pt, float val)
{
	Eigen::Tensor<float, 3> res(pt.dimension(0), pt.dimension(1), pt.dimension(2));
	for (int i = 0; i < pt.dimension(0); ++i)
	{
		for (int j = 0; j < pt.dimension(1); ++j)
		{
			for (int k = 0; k < pt.dimension(2); ++k)
			{
				res(i, j, k) = pt(i, j, k) + val;
			}
		}
	}
	return res;
}


Eigen::Tensor<float, 3> Point_mult(Eigen::Tensor<float, 3> &pt0, Eigen::Tensor<float, 3> &pt1, Eigen::Tensor<float, 3> &pt2)
{
	Eigen::Tensor<float, 3> res(pt0.dimension(0), pt0.dimension(1), pt0.dimension(2));
	for (int i = 0; i < pt0.dimension(0); ++i)
	{
		for (int j = 0; j < pt0.dimension(1); ++j)
		{
			for (int k = 0; k < pt0.dimension(2); ++k)
			{
				res(i, j, k) = pt0(i, j, k) * pt1(i, j, k) * pt2(i, j, k);
			}
		}
	}
	return res;
}

// 13, 12, 9
Eigen::Tensor<float,3> Linwindows(int rayLen, int raydir0, int raydir1)
{
	Eigen::Vector3f dir;
	Eigen::VectorXf scale(2 * (rayLen - 2));

	for (int i = 0; i < 2 * (rayLen - 2); ++i)
	{
		scale(i) = 0.5 * ((2 - rayLen) + i * 2 * (rayLen - 2) / 21.0); // - 5.5 to 5.5
	}
	
	Eigen::Tensor<float, 3>	PointSet(raydir0 * raydir1, 2 * (rayLen - 2), 3);
	int k = 0;
	for (int i = 0; i < raydir0; ++i)
	{
		for (int j = 0; j < raydir1; ++j)
		{
			dir(0) = std::cos(2 * i * PI / raydir0) * std::sin(j * PI / raydir1);
			dir(1) = std::sin(2 * i * PI / raydir0) * std::sin(j * PI / raydir1);
			dir(2) = std::cos(j * PI / raydir1);
			Eigen::MatrixXf temp = scale * dir.transpose();
			for (int m = 0; m < temp.rows(); ++m)
			{
				for (int n = 0; n < temp.cols(); ++n)
				{
					PointSet(k, m, n) = temp(m, n);
				}
			}
			k = k + 1;
		}
	}
	return PointSet;
}


Eigen::Tensor<float, 3> change_dim(Eigen::Tensor<float, 3> &ori)
{
	Eigen::Tensor<float, 3> changed(ori.dimension(0), ori.dimension(2), ori.dimension(1));
	for (int i = 0; i < ori.dimension(0); ++i)
	{
		for (int j = 0; j < ori.dimension(1); ++j)
		{
			for (int k = 0; k < ori.dimension(2); ++k)
				changed(i, k, j) = ori(i, j, k);
		}
	}
	return changed;
}


void scatter_add(Eigen::Tensor<float, 3> &vol, Eigen::Tensor<float, 3> &index, Eigen::Tensor<float, 3> &w)
{
	for (int i = 0; i < w.dimension(0); ++i)
	{
		for (int j = 0; j < w.dimension(1); ++j)
		{
			for (int k = 0; k < w.dimension(2); ++k)
			{
				vol(i, j, int(index(i, j, k))) += w(i, j, k);
			}
		}
	}
}


// transfer direction into weight matrix
std::vector<Eigen::Tensor<float, 3>> PointsTovolumes(int box_size, int n_classes, int oversampling, Eigen::Tensor<float, 3> positions)
{
	int batch_size = positions.dimension(0);
	elmentvise_add_direct(positions, 0.5);
	Eigen::Tensor<float, 3> vol(batch_size, n_classes, (box_size * oversampling) *
	                                                   (box_size * oversampling) *
		                                               (box_size * oversampling));
	vol.setZero();
	Eigen::Tensor<float, 3> xyzpoints(positions.dimension(0), positions.dimension(1), positions.dimension(2));
	for (int i = 0; i < positions.dimension(0); ++i)
		for (int j = 0; j < positions.dimension(1); ++j)
			for (int k = 0; k < positions.dimension(2); ++k)
				xyzpoints(i, j, k) = std::floor(positions(i, j, k));

	Eigen::Tensor<float, 3> rxyz = positions - xyzpoints;
	
	Eigen::array<Eigen::DenseIndex, 3> offsets = {0, 0, 0};
	Eigen::array<Eigen::DenseIndex, 3> extents = {xyzpoints.dimension(0), xyzpoints.dimension(1), 1};
	Eigen::Tensor<float, 3> x = xyzpoints.slice(offsets, extents);
	Eigen::Tensor<float, 3> rx = rxyz.slice(offsets, extents);

	offsets.at(2) = 1;
	Eigen::Tensor<float, 3> y = xyzpoints.slice(offsets, extents);
	Eigen::Tensor<float, 3> ry = rxyz.slice(offsets, extents);
	
	offsets.at(2) = 2;
	Eigen::Tensor<float, 3> z = xyzpoints.slice(offsets, extents);
	Eigen::Tensor<float, 3> rz = rxyz.slice(offsets, extents);
	
	for (int dx = 0; dx < 2; ++dx)
	{
		Eigen::Tensor<float, 3> x_, wx;
		if (dx == 0)
		{
			x_ = x;
			Eigen::Tensor<float, 3> T = - rx;
			wx = elmentvise_add(T, 1);
		}
		else
		{
			x_ = elmentvise_add(x, dx);
			wx = rx;
		}
		for (int dy = 0; dy < 2; ++dy)
		{
			Eigen::Tensor<float, 3> y_, wy;
			if (dy == 0)
			{
				y_ = y;
				Eigen::Tensor<float, 3> T = - ry;
				wy = elmentvise_add(T, 1);
			}
			else
			{
				y_ = elmentvise_add(y, dy);
				wy = ry;
			}
			for (int dz = 0; dz < 2; ++dz)
			{
				Eigen::Tensor<float, 3> z_, wz;
				if (dz == 0)
				{
					z_ = z;
					Eigen::Tensor<float, 3> T = - rz;
					wz = elmentvise_add(T, 1);
				}
				else
				{
					z_ = elmentvise_add(z, dz);
					wz = rz;
				}
				Eigen::Tensor<float, 3> w = Point_mult(wx, wy, wz);
				Eigen::Tensor<float, 3> valid(x_.dimension(0), y_.dimension(1), z_.dimension(2));
				for(int i = 0; i < wx.dimension(0); ++i)
					for(int j = 0; j < wx.dimension(1); ++j)
						for (int k = 0; k < wx.dimension(2); ++k)
						{
							if (x_(i, j, k) >= 0 && x_(i, j, k) < oversampling * box_size
								&& y_(i, j, k) >= 0 && y_(i, j, k) < oversampling * box_size
								&& z_(i, j, k) >= 0 && z_(i, j, k) < oversampling * box_size)
								valid(i, j, k) = 1;
							else
								valid(i, j, k) = 0;
						}
				Eigen::Tensor<float, 3> idx = (oversampling * box_size * (oversampling * box_size * z_ + y_) + x_) * valid;
				Eigen::Tensor<float, 3> idx_new = change_dim(idx);
				w = w * valid;
				Eigen::Tensor<float, 3> w_new = change_dim(w);
				scatter_add(vol, idx_new, w_new);
			}
		}
	}
	
	std::vector<Eigen::Tensor<float, 3>>  kernels(108, Eigen::Tensor<float, 3>(13, 13, 13) );
	for (int i = 0; i < 108; ++i)
	{
		for (int m = 0; m < 13; ++m)
		{
			for (int n = 0; n < 13; ++n)
			{
				for (int l = 0; l < 13; ++l)
				{
					kernels[i](m, n, l) = vol(i, 0, m * 13 * 13 + n * 13 + l);
				}
			}
		}
	}
	return kernels;
}


std::vector<Eigen::Vector3f> Coordinates(Eigen::Vector3f dir)
{
	Eigen::Vector3f dir_abs(std::abs(dir(0)), std::abs(dir(1)), std::abs(dir(2)));
	int id;
	dir_abs.minCoeff(&id);
	Eigen::Vector3f dir1 = Eigen::Vector3f::Zero();
	int id0 = (id + 1) % 3;
	int id1 = (id + 2) % 3;
	dir1(id0) = dir(id1);
	dir1(id1) = -dir(id0);

	Eigen::Vector3f dir2 = Eigen::Vector3f::Zero();
	dir2(0) =   dir(1) * dir1(2) - dir(2) * dir1(1);
	dir2(1) = - dir(0) * dir1(2) + dir(2) * dir1(0);
	dir2(2) =   dir(0) * dir1(1) - dir(1) * dir1(0);

	std::vector<Eigen::Vector3f> res;
	Eigen::Vector3f V1 = dir.normalized();
	Eigen::Vector3f V2 = dir1.normalized();
	Eigen::Vector3f V3 = dir2.normalized();

	res.push_back(V1);
	res.push_back(V2);
	res.push_back(V3);
	return res;
}


Eigen::MatrixXf ImgLineSample(Eigen::Vector3f center, Eigen::Vector3f dir, int Num, Voxel_L &Img)
{
	/*double a = dir(0), b = dir(1), c = dir(2);
	dir(2) = a, dir(0) = c;*/
	Eigen::MatrixXf Points = Eigen::MatrixXf::Zero(4, 2 * Num + 1);
	int nx = Img.dim_z, ny = Img.dim_y, nz = Img.dim_x;
	for (int i = 0; i < 2 * Num + 1; ++i)
	{
		Eigen::Vector3f p = center + (i - Num) * dir;
		int valid = 0;
		if (p(0) >= 0 && p(0) < nx - 1 && p(1) >= 0 && p(1) < ny - 1 && p(2) >= 0 && p(2) < nz - 1)
		{
			valid = 1;
		}
		if (valid == 1)
		{
			float v = Img(std::round(p(2)), std::round(double(p(1))), std::round(p(0)));
			Points.col(i) = Eigen::Vector4f(v, p(0), p(1), p(2));
		}
		else
		{
			Points(0, i) = Img(center(2), center(1), center(0));
		}
	}
	return Points;
}


std::vector<Eigen::MatrixXf> ImgLineSampleSet(Eigen::Vector3f center, Eigen::Vector3f dir0, Eigen::Vector3f dir1, Eigen::Vector3f dir2,
	int Num, Voxel_L &img)
{
	Eigen::MatrixXf Point0 = ImgLineSample(center, dir0, Num, img);
	Eigen::MatrixXf Point1 = ImgLineSample(center, dir1, Num, img);
	Eigen::MatrixXf Point2 = ImgLineSample(center, dir2, Num, img);
	std::vector<Eigen::MatrixXf> res;
	res.push_back(Point0);
	res.push_back(Point1);
	res.push_back(Point2);
	return res;
}


int Lineregiongrowing(Eigen::MatrixXf &Points, int index, double thred, bool COUT = false)
{
	int Num = Points.cols();
	int k = 0;
	if (COUT)
	{
		std::cout << "THRED " << thred << std::endl;
		std::cout << Points << std::endl;
	}
	for (int i = 0; i < Num - index; ++i)
	{
		if (COUT)
		{
			std::cout << "I " << Points(0, i + index) << std::endl;
		}
		if (Points(0, i + index) > thred)
			k = k + 1;
		else
			break;
	}
	int k1 = 0;
	for (int i = 0; i < index; ++i)
	{
		if (COUT)
		{
			std::cout << "I " << Points(0, index - i - 1) << std::endl;
		}
		if (Points(0, index - i - 1) > thred)
			k1 = k1 + 1;
		else
			break;
	}
	return k + k1;
}


int Lineregiongrowing(Eigen::MatrixXf &Points, int index, double thred, double up_thred)
{
	int Num = Points.cols();
	int k = 0;
	for (int i = 0; i < Num - index; ++i)
	{
		if (Points(0, i + index) > thred && Points(0, i + index) < up_thred)
			k = k + 1;
		else
			break;
	}
	int k1 = 0;
	for (int i = 0; i < index; ++i)
	{
		if (Points(0, index - i - 1) > thred && Points(0, index - i - 1) < up_thred)
			k1 = k1 + 1;
		else
			break;
	}
	return k + k1;
}


std::vector<int> LineregiongrowingSET_NEW(std::vector<Eigen::MatrixXf> &Points, int Index, double thred, double up_thred)
{
	int num0 = Lineregiongrowing(Points[0], Index, thred, up_thred);
	int num1 = Lineregiongrowing(Points[1], Index, thred, up_thred);
	int num2 = Lineregiongrowing(Points[2], Index, thred, up_thred);
	std::vector<int> nums;
	nums.push_back(num0);
	nums.push_back(num1);
	nums.push_back(num2);
	return nums;
}


std::vector<int> LineregiongrowingSET(std::vector<Eigen::MatrixXf> &Points, int Index, double thred, bool COUT = false)
{
	int num0 = Lineregiongrowing(Points[0], Index, thred, COUT);
	int num1 = Lineregiongrowing(Points[1], Index, thred, COUT);
	int num2 = Lineregiongrowing(Points[2], Index, thred, COUT);
	std::vector<int> nums;
	nums.push_back(num0);
	nums.push_back(num1);
	nums.push_back(num2);
	return nums;
}


void cut_block(Voxel_E &block, std::string save_dir, std::string name)
{
	Voxel_E temp_block(128, 128, 128);
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 4; ++k)
			{
				Eigen::Vector3i min_range(i * 128, j * 128, k * 128);
				std::string save_name = save_dir + name.substr(0, name.size() - 4) + "_" + std::to_string(i) + "_" +
					std::to_string(j) + "_" + std::to_string(k) + ".tif";
				for (int ix = 0; ix < 128; ++ix)
				{
					for (int iy = 0; iy < 128; ++iy)
					{
						for (int iz = 0; iz < 128; ++iz)
						{
							temp_block(ix, iy, iz) = block(ix + min_range(0), iy + min_range(1), iz + min_range(2));
						}
					}
				}
				temp_block.write_compressed_data(save_name);
			}
		}
	}
}

void cut_block_with_overlap(Voxel_L &block, std::string save_dir, std::string name)
{
	Voxel_L temp_block(192, 192, 192);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				Eigen::Vector3i min_range(i * 160, j * 160, k * 160);
				std::string save_name = save_dir + name.substr(0, name.size() - 4) + "_" + std::to_string(i) + "_" +
					std::to_string(j) + "_" + std::to_string(k) + ".tif";
				for (int ix = 0; ix < 192; ++ix)
				{
					for (int iy = 0; iy < 192; ++iy)
					{
						for (int iz = 0; iz < 192; ++iz)
						{
							temp_block(ix, iy, iz) = block(ix + min_range(0), iy + min_range(1), iz + min_range(2));
						}
					}
				}
				temp_block.write_compressed_data(save_name);
			}
		}
	}
}


void cut_block_with_overlap2(Voxel_L &block, std::string save_dir, std::string name)
{
	int maxx = (block.dim_x - 192.1) / 160.0 + 2;
	int maxy = (block.dim_y - 192.1) / 160.0 + 2;
	int maxz = (block.dim_z - 192.1) / 160.0 + 2;

	Voxel_L temp_block(192, 192, 192);
	for (int i = 0; i < maxx; ++i)
	{
		for (int j = 0; j < maxy; ++j)
		{
			for (int k = 0; k < maxz; ++k)
			{
				Eigen::Vector3i min_range(i * 160, j * 160, k * 160);
				Eigen::Vector3i max_range = {192, 192, 192};
				if (i == maxx - 1)
					max_range(0) = block.dim_x - i * 160;
				if (j == maxy - 1)
					max_range(1) = block.dim_y - j * 160;
				if (k == maxz - 1)
					max_range(2) = block.dim_z - k * 160;
				
				
				std::string save_name = save_dir + name.substr(0, name.size() - 4) + "_" + std::to_string(i) + "_" +
					std::to_string(j) + "_" + std::to_string(k) + ".tif";
				for (int ix = 0; ix < 192; ++ix)
				{
					for (int iy = 0; iy < 192; ++iy)
					{
						for (int iz = 0; iz < 192; ++iz)
						{
							if (ix < max_range(0) && iy < max_range(1) && iz < max_range(2))
								temp_block(ix, iy, iz) = block(ix + min_range(0), iy + min_range(1), iz + min_range(2));
							else
								temp_block(ix, iy, iz) = 0.0;
						}
					}
				}
				temp_block.write_compressed_data(save_name);
			}
		}
	}
}


void generate_gt()
{
	std::string src_dir = "E:/Segmentation/Datasets/QH/raw/";
	std::vector<std::string> files;
	CL_IO::obtain_list(src_dir, files);
	for (int i = 0; i < files.size(); ++i)
	{
		Voxel_L voxel;
		load_tif(src_dir + files[i], voxel);
		cut_block_with_overlap(voxel, "E:/Segmentation/Datasets/QH/raw/cutted/", files[i]);
	}
}


void cut_block(Voxel_L &block, std::string save_dir, std::string name)
{
	Voxel_L temp_block(128, 128, 128);
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 4; ++k)
			{
				Eigen::Vector3i min_range(i * 128, j * 128, k * 128);
				std::string save_name = save_dir + name.substr(0, name.size() - 4) + "_" + std::to_string(i) + "_" +
					std::to_string(j) + "_" + std::to_string(k) + ".tif";
				for (int ix = 0; ix < 128; ++ix)
				{
					for (int iy = 0; iy < 128; ++iy)
					{
						for (int iz = 0; iz < 128; ++iz)
						{
							temp_block(ix, iy, iz) = block(ix + min_range(0), iy + min_range(1), iz + min_range(2));
						}
					}
				}
				temp_block.write_compressed_data(save_name);
			}
		}
	}
}


std::vector<Eigen::Tensor<float, 3>> Gauss_smooth(std::vector<Eigen::Tensor<float, 3>> &kernels)
{
	const double kernel_size = 1.0;
	const double sigma = 1.0;
	Eigen::Tensor<float, 3> gauss_kernel(3, 3, 3);
	for (int i = -1; i <= 1; ++i)
	{
		for (int j = -1; j <= 1; ++j)
		{
			for (int k = -1; k <= 1; ++k)
			{
				gauss_kernel(i + 1, j + 1, k + 1) = exp(-2 * (i * i + j * j + k * k));	
			}	
		}
	}

	std::vector<Eigen::Tensor<float, 3> > kernels_d;
	test_conv(kernels, gauss_kernel, kernels_d);

	std::vector<Eigen::Tensor<float, 3> > smoothed_kernels(kernels.size(), Eigen::Tensor<float, 3>(13, 13, 13));
	for (int i = 0; i < kernels.size(); ++i)
	{
		Eigen::Tensor<float, 3> temp(13, 13, 13);
		for (int m = 0; m < 13; ++m)
		{
			for (int n = 0; n < 13; ++n)
			{
				for (int l = 0; l < 13; ++l)
				{
					double value = 0.0;
					for (int ii = -1; ii <= 1; ii++)
					{
						for (int jj = -1; jj <= 1; ++jj)
						{
							for (int kk = -1; kk <= 1; ++kk)
							{
								if (m + ii >= 0 && m + ii < 13 && n + jj >= 0 && n + jj < 13 && l + kk >= 0 && l + kk < 13)
								{
								/*	if (m == 0 && n == 5 && l == 5)
									{
										std::cout << " " << kernels[i](m + ii, n + jj, l + kk) << " " << gauss_kernel(ii + 1, jj + 1, kk + 1) << " "
											<< ii + 1 << " " << jj + 1 << " " << kk + 1 << " "
											<< m + ii << " " << n + jj << " " << l + kk << std::endl;
									}*/
									value += gauss_kernel(ii + 1, jj + 1, kk + 1) * kernels[i](m + ii, n + jj, l + kk);
								}
							}
						}
					}
					temp(m, n, l) = value;
					if (std::abs(kernels_d[i](m, n, l) - value) > 0.00001)
					{
						std::cout << i << " " << m << " " << n << " " << l << std::endl;
						std::cout << kernels_d[i](m, n, l) << " " << value << std::endl;
					}
				}
			}
		}
		smoothed_kernels[i] = temp;
	}
	return smoothed_kernels;
	/*std::ifstream fin("E:/Segmentation/smoothed_kernel.txt");
	std::vector<Eigen::VectorXd> kenerl_ss(108, Eigen::VectorXd(13 * 13 * 13));
	for (int i = 0; i < 108; ++i)
	{
		for (int j = 0; j < 13 * 13 * 13; ++j)
		{
			fin >> kenerl_ss[i](j);
		}
	}
	fin.close();

	for (int i = 0; i < kernels.size(); ++i)
	{
		for (int m = 0; m < 13; ++m)
		{
			for (int n = 0; n < 13; ++n)
			{
				for (int l = 0; l < 13; ++l)
				{
					if (std::abs(kenerl_ss[i](l + n * 13 + m * 13 * 13) - smoothed_kernels[i](m, n, l)) > 0.00001)
					{
						std::cout << i << " " << m << " " << n << " " << l << std::endl;
						std::cout << kenerl_ss[i](l + n * 13 + m * 13 * 13) << " " << smoothed_kernels[i](m, n, l) << std::endl;
					}
				}
			}
		}
	}
	std::cout << "XX." << std::endl;
	exit(-2);*/

}

void calc_direction_set( std::vector<Eigen::Tensor<float, 3>> &kernels, Voxel_L &raw_img,
						Voxel_E &direction_set)
{
	for (int i = 0; i < raw_img.dim_x; ++i)
	{
		for (int j = 0; j < raw_img.dim_y; ++j)
		{
			for (int k = 0; k < raw_img.dim_z; ++k)
			{
				int idx = 0;
				double max_value = -100000.0;
				for (int ss = 0; ss < kernels.size(); ++ss)
				{
					double value = 0.0;
					for (int ix = - 6; ix <= 6; ++ix)
					{
						for (int iy = - 6; iy <= 6; ++iy)
						{
							for (int iz = - 6; iz <= 6; ++iz)
							{
								if (i + ix >= 0 && i + ix < raw_img.dim_x
									&& j + iy >= 0 && j + iy < raw_img.dim_y
									&& k + iz >= 0 && k + iz < raw_img.dim_z)
								{
									/*std::cout << "{ " << ix + 6 << " " << iy + 6 << " " << iz + 6 << "}" << std::endl;
									std::cout << "{ " << i + ix << " " << j + iy << " " << k + iz << "}" << std::endl;
									std::cout << kernels[ss](ix + 6, iy + 6, iz + 6) << " " << raw_img(i + ix, j + iy, k + iz) << std::endl;*/
									/*value += kernels[ss](ix + 6, iy + 6, iz + 6) * raw_img(i + ix, j + iy, k + iz);*/
									value += kernels[ss](ix + 6, iy + 6, iz + 6) * raw_img(k + iz, j + iy, i + ix);
								}
							}
						}
					}
					//std::cout << ss << " " << value << std::endl;
					/*exit(-14);*/
					if (value > max_value)
					{
						idx = ss;
						max_value = value;
					}
				}
				//exit(-15);
				direction_set(k, j, i) = idx;
				std::cout << k << " " << int(direction_set(k, j, i)) << std::endl;
			}
			exit(-1);
		}
	}
}

void generate_feature_map2(std::string root, std::string dir, double thred2, std::string prefix)
{
	CL_IO::make_dirs(dir);

	auto start0 = std::chrono::high_resolution_clock::now();
	bool exist_direction(false);
	bool save_direction(false);
	double thred = 0.0;

	std::vector<std::string> names;
	CL_IO::make_dirs(dir + "feature_img");
	CL_IO::make_dirs(dir + "raw");
	CL_IO::make_dirs(dir + "label");

	CL_IO::obtain_list(root + "raw", names, "*.tif");
	int par0(13), par1(12), par2(9);
	Eigen::Tensor<float, 3> ss = Linwindows(par0, par1, par2); // 108 * L * 3
	elmentvise_add_direct(ss, 6.0);  // trans for (6, 6, 6)
	int ss_x = par1 * par2;     // 12 * 9 directions
	int ss_y = 2 * (par0 - 2);  // lengths
	int ss_z = 3;
	auto end0 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed0 = end0 - start0;
	std::cout << "time: " << elapsed0.count() << " ms." << std::endl;

	
		// create kernel of directions.
	std::vector<Eigen::Tensor<float, 3>> kernels = PointsTovolumes(13, 1, 1, ss); // 108 * 13 * 13 * 13
	std::vector<Eigen::Tensor<float, 3>> smoothed_kernels = Gauss_smooth(kernels);
	for (int ii = 0; ii < names.size(); ++ii)
	{
		std::cout << names[ii] << " : " << ii << "/" << names.size() << std::endl;
		/*std::filesystem::copy_file(root + "raw/" + names[ii], dir + "raw/" + prefix + names[ii]);
		std::filesystem::copy_file(root + "label/" + names[ii], dir + "label/" + prefix + names[ii]);*/
		std::cout << "copy done." << std::endl;
		/*auto start3 = std::chrono::high_resolution_clock::now();
		auto start1 = std::chrono::high_resolution_clock::now();*/

		Voxel_L lrImage;
		std::cout << "load " << root + "raw/" + names[ii] << std::endl;
		load_tif(root + "raw/" + names[ii], lrImage);
		std::cout << " succed." << std::endl;
		/*auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
		std::cout << "time load tif: " << elapsed1.count() << " ms." << std::endl;*/

		//auto start2 = std::chrono::high_resolution_clock::now();
		std::vector<Eigen::Tensor<float, 3> > res;
		Eigen::Tensor<unsigned char, 3> idx_res(lrImage.dim_x, lrImage.dim_y, lrImage.dim_z);

		//auto start2_1 = std::chrono::high_resolution_clock::now();
		calc_direction_id(smoothed_kernels, lrImage, res, idx_res);

		//auto end2_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<double, std::milli> elapsed2_1 = end2_1 - start2_1;
		//std::cout << "time calc direction: " << elapsed2_1.count() << " ms." << std::endl;

		const int c = 1, d = lrImage.dim_z, w = lrImage.dim_y, h = lrImage.dim_x;
		for (int level = 0; level < 7; level++)
		{
			int type_id = 4;
			Voxel_E Predict_d2(h, w, d);
			fast_iter2(level, lrImage, idx_res, ss, ss_x, ss_y, ss_z, Predict_d2, type_id, thred, thred2);
			CL_IO::make_dirs(dir + "/feature_img/level_" + std::to_string(level) + "/");
			Predict_d2.write_compressed_data(dir + "/feature_img/level_" + std::to_string(level) + "/"
				+ prefix + names[ii]);
		}
		/*auto end3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
		std::cout << "generate and save time: " << elapsed3.count() << " ms." << std::endl;*/
	}
	std::cout << "Generate done." << std::endl;
}



void generate_feature_map(std::string root, std::string save_dir, double thred2)
{
	auto start0 = std::chrono::high_resolution_clock::now();
	bool exist_direction(false);
	bool save_direction(false);
	double thred = 0.0;

	std::vector<std::string> names;
	CL_IO::make_dirs(save_dir + "/feature_img");
	CL_IO::obtain_list(root, names, "*.tif");
	int par0(13), par1(12), par2(9);
	Eigen::Tensor<float, 3> ss = Linwindows(par0, par1, par2); // 108 * L * 3
	elmentvise_add_direct(ss, 6.0);  // trans for (6, 6, 6)
	int ss_x = par1 * par2;     // 12 * 9 directions
	int ss_y = 2 * (par0 - 2);  // lengths
	int ss_z = 3;
	auto end0 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed0 = end0 - start0;

	if (!exist_direction)
	{
		// create kernel of directions.
		std::vector<Eigen::Tensor<float, 3>> kernels = PointsTovolumes(13, 1, 1, ss); // 108 * 13 * 13 * 13
		std::vector<Eigen::Tensor<float, 3>> smoothed_kernels = Gauss_smooth(kernels);
		for (int ii = 0; ii < names.size(); ++ii)
		{
			std::cout << names[ii] << " : " << ii << "/" << names.size() << std::endl;

			auto start3 = std::chrono::high_resolution_clock::now();
			auto start1 = std::chrono::high_resolution_clock::now();
			
			Voxel_L lrImage;
			load_tif(root + "/" + names[ii], lrImage);
			auto end1 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;

			auto start2 = std::chrono::high_resolution_clock::now();
			std::vector<Eigen::Tensor<float, 3> > res;
			Eigen::Tensor<unsigned char, 3> idx_res(lrImage.dim_x, lrImage.dim_y, lrImage.dim_z);

			auto start2_1 = std::chrono::high_resolution_clock::now();
			calc_direction_id(smoothed_kernels, lrImage, res, idx_res);

			auto end2_1 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> elapsed2_1 = end2_1 - start2_1;

			const int c = 1, d = lrImage.dim_z, w = lrImage.dim_y, h = lrImage.dim_x;
			double sum = 0.0;
			for (int level = 0; level < 7; level++)
			{
				int type_id = 4;
				Voxel_E Predict_d2(h, w, d);
				fast_iter2(level, lrImage, idx_res, ss, ss_x, ss_y, ss_z, Predict_d2, type_id, thred, thred2);
				CL_IO::make_dirs(save_dir + "/feature_img/level_" + std::to_string(level) + "/");

				auto s1 = std::chrono::high_resolution_clock::now();
				Predict_d2.write_compressed_data(save_dir + "/feature_img/level_" + std::to_string(level) + "/"
					/*+ std::to_string(type_id) + "_"*/ + names[ii]);
				auto s2 = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> e0 = s2 - s1;
				sum += e0.count();
				//std::cout << "save time: " << e0.count() << " ms." << std::endl;
			}
			auto end3 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
			std::cout << std::endl;
		}
	}
	else
	{
		
	}
	std::cout << "Generate done." << std::endl;
	
}


void cut_image_block(std::string src_dir, std::string save_dir, std::string tar = "")
{
	std::string root = src_dir;
	std::vector<std::string> names;
	CL_IO::obtain_list(root, names, "*.tif");
	for (int i = 0; i < names.size(); ++i)
	{
		if(tar != "")
			names[i] = tar;
		std::string voxel_name = root + names[i];
		Voxel_L block;
		load_tif(voxel_name, block);
		cut_block_with_overlap2(block, save_dir, names[i]);
		if (tar != "")
			break;
	}
}


void merge_image_blocks()
{
	for (int X = 1; X < 11; ++X)
	{
		std::vector<std::string> names;
		std::string path = "E:/Segmentation/Datasets/DIADEM/Neocortical Layer 1 Axons/Neocortical Layer 1 Axons/Subset 2/Image Stacks/0"
			+ std::to_string(X) + "/";
		if(X >= 10)
			path = "E:/Segmentation/Datasets/DIADEM/Neocortical Layer 1 Axons/Neocortical Layer 1 Axons/Subset 2/Image Stacks/"
			+ std::to_string(X) + "/";
		CL_IO::obtain_list(path, names);
		Voxel_L whole_voxel(512, 512, 128);
		for (int i = 0; i < names.size(); ++i)
		{
			std::string name;
			if(i < 9)
				name = "0" + std::to_string(i + 1) + ".tif";
			else
				name = std::to_string(i + 1) + ".tif";
			Voxel_L voxel;
			load_tif(path + name, voxel);
			for (int m = 0; m < 512; ++m)
			{
				for (int n = 0; n < 512; ++n)
				{
					whole_voxel(m, n, i) = voxel(m, n, 0);
				}
			}
		}
		whole_voxel.write_compressed_data("E:/Segmentation/Datasets/DIADEM/Neocortical Layer 1 Axons/raw/2X"
			+ std::to_string(X) + ".tif");
	}
}


int main(int argc, char *argv[]) // generate feature
{
	std::string src_dir;
	std::string save_dir;
	double lambda = 0.4;
	if (argc > 3)
	{
		src_dir = argv[1];
		save_dir = argv[2];
		lambda = std::stod(argv[3]);
		std::cout << src_dir << " " << save_dir << " " << lambda << std::endl;
	}
	else
	{
		return -1;
	}
	generate_feature_map(src_dir, save_dir, lambda);
	return 1;
}


