#include "utils.h"
#include <map>
#include <CL_IO/cl_kdtree.h>
#include <sstream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <CL_IO/cl_io.h>

bool load_tif(std::string file_name, Voxel_L &voxel) {
	TIFF *in = TIFFOpen(file_name.c_str(), "r");
	if (!in) {
		return false;
	}

	uint16 bitspersample = 16;
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

std::vector<std::pair<int, int>> count_change_ratio(Voxel_L &img)
{
	std::vector<std::pair<int, int>> trans;
	for (int x = 1; x < img.dim_x - 1; ++x)
	{
		for (int y = 1; y < img.dim_y - 1; ++y)
		{
			for (int z = 1; z < img.dim_z - 1; ++z)
			{
				Eigen::VectorXd res(27);
				for (int m = 0; m < 3; ++m)
				{
					for (int n = 0; n < 3; ++n)
					{
						for (int k = 0; k < 3; ++k)
						{
							res(k + 3 * n + 9 * m) = std::abs(img(x, y, z) - img(x + m - 1, y + n - 1, z + k - 1));
						}
					}
				}
				int max_dif = res.maxCoeff();
				trans.push_back(std::make_pair(max_dif, int(img(x, y, z))));
			}
		}
	}
	std::sort(trans.begin(), trans.end(), [](const auto& a, const auto& b) {
		return a.second > b.second; // 按 value 升序排序
		});
	return trans;
}

bool voxel2pts(Voxel_L &voxel, Eigen::MatrixXd &pts)
{
	std::vector<double> pts_v;
	for (int x = 0; x < voxel.dim_x; ++x)
	{
		for (int y = 0; y < voxel.dim_y; ++y)
		{
			for (int z = 0; z < voxel.dim_z; ++z)
			{
				if (voxel(x, y, z) > 0)
				{
					pts_v.push_back(x);
					pts_v.push_back(y);
					pts_v.push_back(z);
				}
			}
		}
	}
	pts.resize(3, pts_v.size() / 3);
	memcpy(pts.data(), &pts_v[0], sizeof(double) * pts_v.size());
	return true;
}

void check_feature_map()
{
	std::string root = "E:/Segmentation/TEST_DATASET/SELECT_TEST/";
	std::ofstream fout(root + "modified.txt");
	std::vector<Voxel_L> features(7);
	std::string name = "2324-0019-28_17_30-0_2_0.tif";
	for (int i = 0; i < 7; ++i)
	{
		load_tif(root + "feature_img/level_" + std::to_string(i) + "/" + name, features[i]);
	}
	Voxel_L voxel_seg, voxel_gt, raw;
	std::string gt_dir = root + "mask/";
	std::string seg_dir = root + "test_res_with_feature_no_raw/";
	load_tif(seg_dir + CL_IO::split(name, '.')[0] + "with.tif", voxel_seg);
	load_tif(gt_dir +name, voxel_gt);
	load_tif(root + "raw/" + name, raw);

	for (int x = 0; x < voxel_seg.dim_x; ++x)
	{
		for (int y = 0; y < voxel_seg.dim_y; ++y)
		{
			for (int z = 0; z < voxel_seg.dim_z; ++z)
			{
				if (voxel_gt(x, y, z) == 0)
				{
					/*if (voxel_seg(x, y, z) == 0)
					{
						std::cout << " RIGHT BACK { " << x << " " << y << " " << z << " ||| "
							<< raw(x, y, z) << " |||| ";
						for (int i = 0; i < 7; ++i)
							std::cout << features[i](x, y, z) << " ";
						std::cout << " }";
						std::cout << std::endl;
					}
					else
					{
						std::cout << " WRONG BACK { " << x << " " << y << " " << z << " ||| "
							<< raw(x, y, z) << " |||| ";
						for (int i = 0; i < 7; ++i)
							std::cout << features[i](x, y, z) << " ";
						std::cout << " }";
						std::cout << std::endl;
					}*/
					;
				}
				else
				{
					if (voxel_seg(x, y, z) != 0)
					{
						std::cout << " RIGHT FORE { " << x << " " << y << " " << z << " ||| " 
							<< raw(x, y, z) << " |||| ";
						for (int i = 0; i < 7; ++i)
							std::cout << features[i](x, y, z) << " ";
						std::cout << " }";
						std::cout << std::endl;
					}
					else
					{
						fout << x << " " << y << " " << z << std::endl;
						/*std::cout << " WRONG FORE { " << x << " " << y << " " << z << " ||| "
							<< raw(x, y, z) << " |||| ";
						for (int i = 0; i < 7; ++i)
							std::cout << features[i](x, y, z) << " ";
						std::cout << " }";
						std::cout << std::endl;*/
						features[0](x, y, z) = 251;
						features[1](x, y, z) = 251;
						features[2](x, y, z) = 251;
						features[3](x, y, z) = 248;
						features[4](x, y, z) = 248;
						features[5](x, y, z) = 248;
						features[6](x, y, z) = 245;
					}
				}

			}
		}
	}
	for (int i = 0; i < 7; ++i)
		features[i].write_compressed_data(root + "feature_img_m/level_" + std::to_string(i) + "/2324-0019-28_17_30-0_2_0.tif");

}


int compare_two_skeletons(const Eigen::MatrixXd &pt0, const Eigen::MatrixXd &gt, std::vector<int> &R_G_TP) {
	// std::cout << pt0.rows() << " " << pt0.cols() << " " << gt.rows() << " " << gt.cols() << std::endl;
	KDTree *tree_src = new KDTree;
	KDTree *tree_tar = new KDTree;
	tree_src->build_tree(pt0, 3);
	tree_tar->build_tree(gt, 3);

	int S_G = gt.cols();
	int S_extra = 0;
	int num_tp = 0;
	int num_tp2 = 0;
	Res res(1);
	double thred = 2.0;

	for (int i = 0; i < pt0.cols(); ++i) {
		tree_tar->search_pts(pt0.col(i), res);

		if (res.dists(0) > thred * thred )
			S_extra += 1;
		if (res.dists(0) <= thred * thred)
			num_tp += 1;
	}

	int S_miss = 0;
	for (int i = 0; i < gt.cols(); ++i) {
		tree_src->search_pts(gt.col(i), res);
		if (res.dists(0) > thred * thred)
			S_miss += 1;
		else
			num_tp2 += 1;
	}
	R_G_TP.push_back(pt0.cols());
	R_G_TP.push_back(gt.cols());
	R_G_TP.push_back(num_tp);
	
	R_G_TP.push_back(num_tp2);
	/*valid_num = S_G - S_miss;*/
	delete tree_src;
	delete tree_tar;
	return num_tp;
}


void calc_score(std::string dir, std::vector<std::string> name_list, std::vector<std::string> sub_fixs,
	std::vector<std::string> corres_fixs)
{
	std::ofstream fout, fout2;
	std::string out_name = dir + "compare";
	std::string out_name2 = dir + "only";

	for (int i = 0; i < sub_fixs.size(); ++i)
	{
		out_name += "_" + sub_fixs[i];
		out_name2 += "_" + sub_fixs[i];
	}
	out_name += ".txt";
	out_name2 += ".txt";
	
	fout.open(out_name);
	fout2.open(out_name2);

	double average_precsion_with = 0.0;
	double average_recall_with = 0.0;
	double average_precision_without = 0.0;
	double average_recall_without = 0.0;


	double average_precsion_with_new = 0.0;
	double average_recall_with_new = 0.0;
	double average_f1_with_new = 0.0;
	double average_precision_without_new = 0.0;
	double average_recall_without_new = 0.0;
	double average_f1_without_new = 0.0;

	int number_with = 0;
	int number_without = 0;

	for (int i = 0; i < name_list.size(); ++i)
	{
		std::vector<std::string> suffix = sub_fixs;
		std::string gt_dir = dir + "mask/";
		Voxel_L voxel_gt;
		load_tif(gt_dir + name_list[i], voxel_gt);

		Eigen::MatrixXd gt_pts;
		voxel2pts(voxel_gt, gt_pts);
		std::cout << gt_pts.cols() << std::endl;
		KDTree *gt_tree;

		fout << name_list[i];
		for (int id = 0; id < suffix.size(); ++id)
		{
			Voxel_L voxel_seg;	
			std::string seg_dir = dir + suffix[id] + "/";
			
			if (!load_tif(seg_dir + CL_IO::split(name_list[i], '.')[0] + corres_fixs[id] + ".tif", voxel_seg))
				continue;
			

			Eigen::MatrixXd seg_pts;
			voxel2pts(voxel_seg, seg_pts);
			
			if (seg_pts.cols() == 0 || gt_pts.cols() == 0)
				continue;
			std::vector<int> TPS;
			compare_two_skeletons(seg_pts, gt_pts, TPS);
			double precision_new = double(TPS[2]) / TPS[0];
			double recall_new = double(TPS[3]) / TPS[1];
			double f1_new = 2 * precision_new * recall_new / (precision_new + recall_new);

			int TP = 0, FP = 0, TN = 0, FN = 0;
			for (int x = 0; x < voxel_seg.dim_x; ++x)
			{
				for (int y = 0; y < voxel_seg.dim_y; ++y)
				{
					for (int z = 0; z < voxel_seg.dim_z; ++z)
					{
						if (voxel_gt(x, y, z) == 0)
						{
							if (voxel_seg(x, y, z) == 0)
								TN += 1;
							else
								FP += 1;
						}
						else
						{
							if (voxel_seg(x, y, z) != 0)
								TP += 1;
							else
							{
								FN += 1;
							}
						}

					}
				}
			}
			double precision = double(TP) / (TP + FP);
			double recall = double(TP) / (TP + FN);
			if (id == 0)
			{
				average_precsion_with += precision;
				average_recall_with += recall;
				average_precsion_with_new += precision_new;
				average_recall_with_new += recall_new;
				average_f1_with_new += f1_new;
			}
			else
			{
				average_precision_without += precision;
				average_recall_without += recall;
				average_precision_without_new += precision_new;
				average_recall_without_new += recall_new;
				average_f1_without_new += f1_new;
			}
			if (id == 0)
				number_with += 1;
			else
				number_without += 1;
			std::cout << name_list[i] << std::endl;
			std::cout << "Precision: " << precision << " Recall: " << recall << std::endl;
			std::cout << "New Precision: " << precision_new << " Recall: " << recall_new << " f1: " << f1_new << " " << average_f1_with_new 
				 << " " << average_f1_without_new << std::endl;
			std::cout << suffix[id] << " TP " << TP << " FP " << FP << " TN " << TN << " FN " << FN << std::endl;
			std::cout << std::endl;
			fout << " " << precision_new << " " << recall_new << " " << f1_new << " ||| | ";
			fout2 << f1_new << " ";
		}
		fout << std::endl;
		fout2 << std::endl;
	}
	
	std::cout << "average with precision: " << average_precsion_with / number_with << " recall: " << average_recall_with / number_with << std::endl;
	std::cout << "average without precision: " << average_precision_without / number_without << " recall: " << average_recall_without / number_without << std::endl;
	std::cout << "new average with precision: " << average_precsion_with_new / number_with << " recall: " << average_recall_with_new / number_with 
		<< " f1: " << average_f1_with_new / number_with << std::endl;
	std::cout << "new average without precision: " << average_precision_without_new / number_without << " recall: " << average_recall_without_new / number_without
		<< " f1: " << average_f1_without_new / number_without << std::endl;
	std::cout << average_f1_without_new << " " << number_without << std::endl;

	fout << "average with precision: " << average_precsion_with / number_with << " recall: " << average_recall_with / number_with << std::endl;
	fout << "average without precision: " << average_precision_without / number_without << " recall: " << average_recall_without / number_without << std::endl;
	fout << "new average with precision: " << average_precsion_with_new / number_with << " recall: " << average_recall_with_new / number_with 
		 <<  " f1: " << average_f1_with_new / number_with << std::endl;
	fout << "new average without precision: " << average_precision_without_new / number_without << " recall: " << average_recall_without_new / number_without
		 << " f1: " << average_f1_without_new / number_without << std::endl;
	
}


bool isNumericLine(const std::string& line) {
	for (char c : line) {
		// 允许的字符：数字、小数点、空格、负号、制表符
		if (!(isdigit(c) || c == '.' || c == ' ' || c == '-' || c == '\t')) {
			return false;
		}
	}
	return true;
}


void trans_mra_tre(std::string root, std::string name)
{
	std::ifstream fin(root + name);
	std::string line;
	std::ofstream fout(root + CL_IO::split(name, '.')[0] + ".swc");
	int id = 1;
	bool newline = true;
	while (std::getline(fin, line))
	{
		
		if (isNumericLine(line))
		{
			std::stringstream ss;
			ss << line;
			double x, y, z;
			ss >> x >> y >> z;
			if (id == 1 || newline)
				fout << id << " 0 " << x << " " << y << " " << z << " 0 -1" << std::endl;
			else
				fout << id << " 0 " << x << " " << y << " " << z << " 0 " << id - 1 << std::endl;
			id++;
			newline = false;
		}
		else
			newline = true;

	}
	fout.close();

}


void generate_diadem_swc(std::string src_swc_root, int swc_number)
{
	std::vector<Eigen::MatrixXd> swc_datas(swc_number);
	for (int i = 1; i < swc_number + 1; ++i)
	{
		std::string name;
		if(i < 10)
			 name = src_swc_root + "NC_0" + std::to_string(i) + ".swc";
		else
			name = src_swc_root + "NC_" + std::to_string(i) + ".swc";
		Eigen::MatrixXd swc_data;
		CL_IO::load_swc(name, swc_datas[i]);
	}
	std::vector<Eigen::Vector3d> ranges{Eigen::Vector3d(73, 507, -5), Eigen::Vector3d(526, 484, 11), Eigen::Vector3d(952, 462, -21),
										Eigen::Vector3d(924, 3, -19), Eigen::Vector3d(468, -14, -1), Eigen::Vector3d(0, 0, 0) };
	for (int i = 0; i < ranges.size(); ++i)
	{
		std::ofstream fout("E:/Segmentation/Datasets/DIADEM/Neocortical Layer 1 Axons/raw/1X" + std::to_string(i + 1) + ".swc");
		int id = 1;
		for (int j = 0; j < swc_datas.size(); ++j)
		{
			bool last = false;
			for (int k = 0; k < swc_datas[j].cols(); ++k)
			{
				std::cout << swc_datas[j](2, k) << " " << swc_datas[j](3, k) << " " << swc_datas[j](4, k) << std::endl;
				if (check_is_in_range(Eigen::Vector3d(swc_datas[j](2, k), swc_datas[j](3, k), swc_datas[j](4, k)),
					ranges[i], ranges[i] + Eigen::Vector3d(512, 512, 128)))
				{
					if (last == false)
						fout << id << " 0 " << swc_datas[j](2, k) - ranges[i](0) << " " << swc_datas[j](3, k) - ranges[i](1) << " " << swc_datas[j](4, k)
						- ranges[i](2) << " 0 -1" << std::endl;
					else
						fout << id << " 0 " << swc_datas[j](2, k) - ranges[i](0) << " " << swc_datas[j](3, k) - ranges[i](1) << " " << swc_datas[j](4, k)
						- ranges[i](2) << " 0 " << id - 1 << std::endl;
					id++;
					last = true;
				}
				else
					last = false;
			}
		}
		fout.close();
	}
}


bool check_is_in_range(Eigen::Vector3d pt, Eigen::Vector3d range0, Eigen::Vector3d range1)
{
	if (pt(0) >= range0(0) && pt(0) <= range1(0) &&
		pt(1) >= range0(1) && pt(1) <= range1(1) && 
		pt(2) >= range0(2) && pt(2) <= range1(2))
	{
		return true;
	}
	return false;
}


void combine_diadem_gt(std::string root, std::string save_name)
{
	int id = 0;
	std::ofstream fout(save_name);
	for (char i = 'A'; i <= 'U'; ++i)
	{
		std::string name;
		//if (i < 10)
		name = root + "/NC_" + i + ".swc";
		//else
			//name = root + "/NC_" + std::to_string(i) + ".swc";
		Eigen::MatrixXd swc_data;
		CL_IO::load_swc(name, swc_data);

		for (int j = 0; j < swc_data.cols(); ++j)
		{
			if (swc_data(6, j) == -1)
				fout << swc_data(0, j) + id << " " << swc_data(1, j) <<
					" " << swc_data(2, j) << " " << swc_data(3, j) + 0
					<< " " << swc_data(4, j) + 22 << " 0 -1" << std::endl;
			else
				fout << swc_data(0, j) + id << " " << swc_data(1, j) <<
				" " << swc_data(2, j) << " " << swc_data(3, j) + 0
				<< " " << swc_data(4, j) + 22 << " 0 " << swc_data(6, j) + id << std::endl;
		}
		id += swc_data.cols();
	}
	fout.close();
}


//Eigen::Tensor<float, 3> Linwindows2(int rayLen, int raydir0, int raydir1)
//{
//	double PI = 3.1415926;
//	Eigen::Vector3f dir;
//	Eigen::VectorXf scale(2 * (rayLen - 2));
//
//	for (int i = 0; i < 2 * (rayLen - 2); ++i)
//	{
//		scale(i) = 0.5 * ((2 - rayLen) + i * 2 * (rayLen - 2) / 21.0); // - 5.5 to 5.5
//	}
//
//	Eigen::Tensor<float, 3>	PointSet(raydir0 * raydir1, 2 * (rayLen - 2), 3);
//	int k = 0;
//	for (int i = 0; i < raydir0; ++i)
//	{
//		for (int j = 0; j < raydir1; ++j)
//		{
//			dir(0) = std::cos(2 * i * PI / raydir0) * std::sin(j * PI / raydir1);
//			dir(1) = std::sin(2 * i * PI / raydir0) * std::sin(j * PI / raydir1);
//			dir(2) = std::cos(j * PI / raydir1);
//			Eigen::MatrixXf temp = scale * dir.transpose();
//			for (int m = 0; m < temp.rows(); ++m)
//			{
//				for (int n = 0; n < temp.cols(); ++n)
//				{
//					PointSet(k, m, n) = temp(m, n);
//				}
//			}
//			k = k + 1;
//		}
//	}
//	return PointSet;
//}
//
//
//void elmentvise_add_direct2(Eigen::Tensor<float, 3> &pt, float val)
//{
//	for (int i = 0; i < pt.dimension(0); ++i)
//	{
//		for (int j = 0; j < pt.dimension(1); ++j)
//		{
//			for (int k = 0; k < pt.dimension(2); ++k)
//			{
//				pt(i, j, k) += val;
//			}
//		}
//	}
//}
//
//
//Eigen::Tensor<float, 3> elmentvise_add2(Eigen::Tensor<float, 3> &pt, float val)
//{
//	Eigen::Tensor<float, 3> res(pt.dimension(0), pt.dimension(1), pt.dimension(2));
//	for (int i = 0; i < pt.dimension(0); ++i)
//	{
//		for (int j = 0; j < pt.dimension(1); ++j)
//		{
//			for (int k = 0; k < pt.dimension(2); ++k)
//			{
//				res(i, j, k) = pt(i, j, k) + val;
//			}
//		}
//	}
//	return res;
//}
//
//
//Eigen::Tensor<float, 3> change_dim2(Eigen::Tensor<float, 3> &ori)
//{
//	Eigen::Tensor<float, 3> changed(ori.dimension(0), ori.dimension(2), ori.dimension(1));
//	for (int i = 0; i < ori.dimension(0); ++i)
//	{
//		for (int j = 0; j < ori.dimension(1); ++j)
//		{
//			for (int k = 0; k < ori.dimension(2); ++k)
//				changed(i, k, j) = ori(i, j, k);
//		}
//	}
//	return changed;
//}
//
//
//Eigen::Tensor<float, 3> Point_mult2(Eigen::Tensor<float, 3> &pt0, Eigen::Tensor<float, 3> &pt1, Eigen::Tensor<float, 3> &pt2)
//{
//	Eigen::Tensor<float, 3> res(pt0.dimension(0), pt0.dimension(1), pt0.dimension(2));
//	for (int i = 0; i < pt0.dimension(0); ++i)
//	{
//		for (int j = 0; j < pt0.dimension(1); ++j)
//		{
//			for (int k = 0; k < pt0.dimension(2); ++k)
//			{
//				res(i, j, k) = pt0(i, j, k) * pt1(i, j, k) * pt2(i, j, k);
//			}
//		}
//	}
//	return res;
//}
//
//
//void scatter_add2(Eigen::Tensor<float, 3> &vol, Eigen::Tensor<float, 3> &index, Eigen::Tensor<float, 3> &w)
//{
//	for (int i = 0; i < w.dimension(0); ++i)
//	{
//		for (int j = 0; j < w.dimension(1); ++j)
//		{
//			for (int k = 0; k < w.dimension(2); ++k)
//			{
//				vol(i, j, int(index(i, j, k))) += w(i, j, k);
//			}
//		}
//	}
//}
//
//
//// transfer direction into weight matrix
//std::vector<Eigen::Tensor<float, 3>> PointsTovolumes2(int box_size, int n_classes, int oversampling, Eigen::Tensor<float, 3> positions)
//{
//	int batch_size = positions.dimension(0);
//	elmentvise_add2(positions, 0.5);
//	Eigen::Tensor<float, 3> vol(batch_size, n_classes, (box_size * oversampling) *
//		(box_size * oversampling) *
//		(box_size * oversampling));
//	vol.setZero();
//	Eigen::Tensor<float, 3> xyzpoints(positions.dimension(0), positions.dimension(1), positions.dimension(2));
//	for (int i = 0; i < positions.dimension(0); ++i)
//		for (int j = 0; j < positions.dimension(1); ++j)
//			for (int k = 0; k < positions.dimension(2); ++k)
//				xyzpoints(i, j, k) = std::floor(positions(i, j, k));
//
//	Eigen::Tensor<float, 3> rxyz = positions - xyzpoints;
//
//	Eigen::array<Eigen::DenseIndex, 3> offsets = { 0, 0, 0 };
//	Eigen::array<Eigen::DenseIndex, 3> extents = { xyzpoints.dimension(0), xyzpoints.dimension(1), 1 };
//	Eigen::Tensor<float, 3> x = xyzpoints.slice(offsets, extents);
//	Eigen::Tensor<float, 3> rx = rxyz.slice(offsets, extents);
//
//	offsets.at(2) = 1;
//	Eigen::Tensor<float, 3> y = xyzpoints.slice(offsets, extents);
//	Eigen::Tensor<float, 3> ry = rxyz.slice(offsets, extents);
//
//	offsets.at(2) = 2;
//	Eigen::Tensor<float, 3> z = xyzpoints.slice(offsets, extents);
//	Eigen::Tensor<float, 3> rz = rxyz.slice(offsets, extents);
//
//	for (int dx = 0; dx < 2; ++dx)
//	{
//		Eigen::Tensor<float, 3> x_, wx;
//		if (dx == 0)
//		{
//			x_ = x;
//			Eigen::Tensor<float, 3> T = -rx;
//			wx = elmentvise_add2(T, 1);
//		}
//		else
//		{
//			x_ = elmentvise_add2(x, dx);
//			wx = rx;
//		}
//		for (int dy = 0; dy < 2; ++dy)
//		{
//			Eigen::Tensor<float, 3> y_, wy;
//			if (dy == 0)
//			{
//				y_ = y;
//				Eigen::Tensor<float, 3> T = -ry;
//				wy = elmentvise_add2(T, 1);
//			}
//			else
//			{
//				y_ = elmentvise_add2(y, dy);
//				wy = ry;
//			}
//			for (int dz = 0; dz < 2; ++dz)
//			{
//				Eigen::Tensor<float, 3> z_, wz;
//				if (dz == 0)
//				{
//					z_ = z;
//					Eigen::Tensor<float, 3> T = -rz;
//					wz = elmentvise_add2(T, 1);
//				}
//				else
//				{
//					z_ = elmentvise_add2(z, dz);
//					wz = rz;
//				}
//				Eigen::Tensor<float, 3> w = Point_mult2(wx, wy, wz);
//				Eigen::Tensor<float, 3> valid(x_.dimension(0), y_.dimension(1), z_.dimension(2));
//				for (int i = 0; i < wx.dimension(0); ++i)
//					for (int j = 0; j < wx.dimension(1); ++j)
//						for (int k = 0; k < wx.dimension(2); ++k)
//						{
//							if (x_(i, j, k) >= 0 && x_(i, j, k) < oversampling * box_size
//								&& y_(i, j, k) >= 0 && y_(i, j, k) < oversampling * box_size
//								&& z_(i, j, k) >= 0 && z_(i, j, k) < oversampling * box_size)
//								valid(i, j, k) = 1;
//							else
//								valid(i, j, k) = 0;
//						}
//				Eigen::Tensor<float, 3> idx = (oversampling * box_size * (oversampling * box_size * z_ + y_) + x_) * valid;
//				Eigen::Tensor<float, 3> idx_new = change_dim2(idx);
//				w = w * valid;
//				Eigen::Tensor<float, 3> w_new = change_dim2(w);
//				scatter_add2(vol, idx_new, w_new);
//			}
//		}
//	}
//
//	std::vector<Eigen::Tensor<float, 3>>  kernels(108, Eigen::Tensor<float, 3>(13, 13, 13));
//	for (int i = 0; i < 108; ++i)
//	{
//		for (int m = 0; m < 13; ++m)
//		{
//			for (int n = 0; n < 13; ++n)
//			{
//				for (int l = 0; l < 13; ++l)
//				{
//					kernels[i](m, n, l) = vol(i, 0, m * 13 * 13 + n * 13 + l);
//				}
//			}
//		}
//	}
//	return kernels;
//}
//
//
//std::vector<Eigen::Tensor<float, 3>> Gauss_smooth2(std::vector<Eigen::Tensor<float, 3>> &kernels)
//{
//	const double kernel_size = 1.0;
//	const double sigma = 1.0;
//	Eigen::Tensor<float, 3> gauss_kernel(3, 3, 3);
//	for (int i = -1; i <= 1; ++i)
//	{
//		for (int j = -1; j <= 1; ++j)
//		{
//			for (int k = -1; k <= 1; ++k)
//			{
//				gauss_kernel(i + 1, j + 1, k + 1) = exp(-2 * (i * i + j * j + k * k));
//			}
//		}
//	}
//
//	std::vector<Eigen::Tensor<float, 3> > kernels_d;
//	test_conv(kernels, gauss_kernel, kernels_d);
//
//	std::vector<Eigen::Tensor<float, 3> > smoothed_kernels(kernels.size(), Eigen::Tensor<float, 3>(13, 13, 13));
//	for (int i = 0; i < kernels.size(); ++i)
//	{
//		Eigen::Tensor<float, 3> temp(13, 13, 13);
//		for (int m = 0; m < 13; ++m)
//		{
//			for (int n = 0; n < 13; ++n)
//			{
//				for (int l = 0; l < 13; ++l)
//				{
//					double value = 0.0;
//					for (int ii = -1; ii <= 1; ii++)
//					{
//						for (int jj = -1; jj <= 1; ++jj)
//						{
//							for (int kk = -1; kk <= 1; ++kk)
//							{
//								if (m + ii >= 0 && m + ii < 13 && n + jj >= 0 && n + jj < 13 && l + kk >= 0 && l + kk < 13)
//								{
//									/*	if (m == 0 && n == 5 && l == 5)
//										{
//											std::cout << " " << kernels[i](m + ii, n + jj, l + kk) << " " << gauss_kernel(ii + 1, jj + 1, kk + 1) << " "
//												<< ii + 1 << " " << jj + 1 << " " << kk + 1 << " "
//												<< m + ii << " " << n + jj << " " << l + kk << std::endl;
//										}*/
//									value += gauss_kernel(ii + 1, jj + 1, kk + 1) * kernels[i](m + ii, n + jj, l + kk);
//								}
//							}
//						}
//					}
//					temp(m, n, l) = value;
//					if (std::abs(kernels_d[i](m, n, l) - value) > 0.00001)
//					{
//						std::cout << i << " " << m << " " << n << " " << l << std::endl;
//						std::cout << kernels_d[i](m, n, l) << " " << value << std::endl;
//					}
//				}
//			}
//		}
//		smoothed_kernels[i] = temp;
//	}
//	return smoothed_kernels;
//}
//
//
//void generate_feature_map2(std::string root, std::string tar, double thred2)
//{
//	auto start0 = std::chrono::high_resolution_clock::now();
//	bool exist_direction(false);
//	bool save_direction(false);
//	double thred = 0.0;
//
//	std::vector<std::string> names;
//	CL_IO::make_dirs(root + "/feature_img");
//	CL_IO::obtain_list(root + "/raw", names, "*.tif");
//	int par0(13), par1(12), par2(9);
//	Eigen::Tensor<float, 3> ss = Linwindows2(par0, par1, par2); // 108 * L * 3
//	elmentvise_add2(ss, 6.0);  // trans for (6, 6, 6)
//	int ss_x = par1 * par2;     // 12 * 9 directions
//	int ss_y = 2 * (par0 - 2);  // lengths
//	int ss_z = 3;
//	auto end0 = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double, std::milli> elapsed0 = end0 - start0;
//	std::cout << "time: " << elapsed0.count() << " ms." << std::endl;
//
//	if (!exist_direction)
//	{
//		// create kernel of directions.
//		std::vector<Eigen::Tensor<float, 3>> kernels = PointsTovolumes2(13, 1, 1, ss); // 108 * 13 * 13 * 13
//		std::vector<Eigen::Tensor<float, 3>> smoothed_kernels = Gauss_smooth2(kernels);
//		for (int ii = 0; ii < names.size(); ++ii)
//		{
//			/*if (names[ii] != "622-0470-12_17_32-0_2_0.tif")
//				continue;*/
//			std::cout << names[ii] << " : " << ii << "/" << names.size() << std::endl;
//
//			auto start3 = std::chrono::high_resolution_clock::now();
//			auto start1 = std::chrono::high_resolution_clock::now();
//
//			Voxel_L lrImage;
//			load_tif(root + "/raw/" + names[ii], lrImage);
//			auto end1 = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
//			std::cout << "time load tif: " << elapsed1.count() << " ms." << std::endl;
//
//			auto start2 = std::chrono::high_resolution_clock::now();
//			std::vector<Eigen::Tensor<float, 3> > res;
//			Eigen::Tensor<unsigned char, 3> idx_res(lrImage.dim_x, lrImage.dim_y, lrImage.dim_z);
//
//			auto start2_1 = std::chrono::high_resolution_clock::now();
//			calc_direction_id(smoothed_kernels, lrImage, res, idx_res);
//
//			auto end2_1 = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double, std::milli> elapsed2_1 = end2_1 - start2_1;
//			std::cout << "time calc direction: " << elapsed2_1.count() << " ms." << std::endl;
//
//			const int c = 1, d = lrImage.dim_z, w = lrImage.dim_y, h = lrImage.dim_x;
//			for (int level = 0; level < 7; level++)
//			{
//				int type_id = 4;
//				Voxel_E Predict_d2(h, w, d);
//				fast_iter2(level, lrImage, idx_res, ss, ss_x, ss_y, ss_z, Predict_d2, type_id, thred, thred2);
//				CL_IO::make_dirs(root + "/feature_img/level_" + std::to_string(level) + "/");
//				Predict_d2.write_compressed_data(root + "/feature_img/level_" + std::to_string(level) + "/"
//					/*+ std::to_string(type_id) + "_"*/ + names[ii]);
//			}
//			auto end3 = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
//			std::cout << "generate and save time: " << elapsed3.count() << " ms." << std::endl;
//		}
//	}
//	else
//	{
//	}
//	std::cout << "Generate done." << std::endl;
//
//}
//
//
//void generate_enhanced_features(std::string root, std::string target, std::vector<double> enhanced_para)
//{
//	CL_IO::make_dirs(target);
//	CL_IO::make_dirs(target + "raw/");
//	CL_IO::make_dirs(target + "label/");
//	CL_IO::make_dirs(target + "feature_img/");
//	for (int i = 0; i < enhanced_para.size(); ++i)
//	{
//		generate_feature_map2(root, target, 0.4);
//
//	}
//
//}
