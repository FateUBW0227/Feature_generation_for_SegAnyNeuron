#pragma once
#include "voxel.h"
#include <map>
#include <tiffio.h>
#include <CL_IO/cl_io.h>

// dif, Intensity.
std::vector<std::pair<int, int>> count_change_ratio(Voxel_L &img);

void calc_score(std::string seg_dir, std::vector<std::string> name_list, std::vector<std::string> sub_fixs,
	std::vector<std::string> corres_fix);

bool load_tif(std::string file_name, Voxel_L &voxel);

int compare_two_skeletons(const Eigen::MatrixXd &pt0, const Eigen::MatrixXd &gt, std::vector<int> &R_G_TP);

bool voxel2pts(Voxel_L &voxel, Eigen::MatrixXd &pts);

void check_feature_map();

void trans_mra_tre(std::string root, std::string name);

void generate_diadem_swc(std::string src_swc_root, int swc_number);

bool check_is_in_range(Eigen::Vector3d pt, Eigen::Vector3d range0, Eigen::Vector3d range1);

void generate_enhanced_features(std::string root, std::string target, std::vector<double> enhanced_para);

void combine_diadem_gt(std::string root, std::string save_name);

//void generate_feature_map2(std::string root, std::string tar, double thred2);
//
//Eigen::Tensor<float, 3> Linwindows2(int rayLen, int raydir0, int raydir1);
//
//std::vector<Eigen::Tensor<float, 3>> PointsTovolumes2(int box_size, int n_classes, int oversampling, Eigen::Tensor<float, 3> positions);
//
//Eigen::Tensor<float, 3> elmentvise_add2(Eigen::Tensor<float, 3> &pt, float val);
//
//void elmentvise_add_direct2(Eigen::Tensor<float, 3> &pt, float val);
//
//std::vector<Eigen::Tensor<float, 3>> Gauss_smooth2(std::vector<Eigen::Tensor<float, 3>> &kernels);
//
//Eigen::Tensor<float, 3> change_dim2(Eigen::Tensor<float, 3> &ori);
//
//void scatter_add2(Eigen::Tensor<float, 3> &vol, Eigen::Tensor<float, 3> &index, Eigen::Tensor<float, 3> &w);
//
//Eigen::Tensor<float, 3> Point_mult2(Eigen::Tensor<float, 3> &pt0, Eigen::Tensor<float, 3> &pt1, Eigen::Tensor<float, 3> &pt2);
