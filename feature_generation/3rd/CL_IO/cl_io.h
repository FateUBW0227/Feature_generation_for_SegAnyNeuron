#ifndef CL_IO_H
#define CL_IO_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
#include "Voxel.h"
#include <tiffio.h>
#include <QtCore/QDir>
#include <chrono>
#include <sstream>
//#include <mat.h>

class CL_IO
{
public:
    CL_IO();
    ~CL_IO();

	static bool build_swc_structure(Eigen::MatrixXd &swc_data, std::vector<Eigen::MatrixXd> &structured_swc);

	static int check_file_num(std::string suffix, std::string target_dir);

	static bool copy_file(std::string src_file, std::string tar_file);

    static void COUT(std::string str);

	static double Cout_Time(std::vector<std::chrono::steady_clock::time_point> &markers);

    template <typename T>
    static void COUT_VEC(std::vector<T> array, std::string st = "");

    template <typename T>
    static void COUT_VEC(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> array, std::string st);

	// modify name to dir name.
	static std::string dir(std::string dir_name);

	static double file_size(std::string name);

    template <typename T>
    static bool load_mat(const std::string name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat,
                         int cols, bool block = false);

    template <typename T>
    static bool load_swc(const std::string name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
                  &pts, Eigen::MatrixXd &others, bool block = false);

	template <typename T>
	static bool load_swc(const std::string swc_file, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &swc_data);

    static bool make_dirs(std::string name);

	static bool increase_density(Eigen::MatrixXd &swc_data);

    static bool read_image(std::string file_name, IVoxel &voxel, bool block = true);

    static bool obtain_list(std::string dir_name, std::vector<std::string> &file_name,
                            std::string suffix = "");

    static bool output2matlab(double *data, std::string file_name, std::string variable_name,
                              int row, int col);

	template <typename T>
	static bool save_swc(std::string file_name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &swc_data);

	static bool save_time(std::vector<std::chrono::steady_clock::time_point> &markers);

	static std::vector<Eigen::MatrixXd> split_into_trees(Eigen::MatrixXd &swc_data);

	static bool re_scale_swc(std::string src_file, std::string tar_file, double ratio0, double ratio1, double ratio2);

	static std::vector<std::string> split(std::string str, char character);

    static bool write_image(std::string name, IVoxel &voxel);

    std::vector<Eigen::Vector3i> COLORS;

    // static struct timeval start_time, end_time;
	// static int a;

    static void TIME(int id);
};


__inline CL_IO::CL_IO()
{
    COLORS.resize(20);
    COLORS[0]  = Eigen::Vector3i(0,   0,   0);
    COLORS[1]  = Eigen::Vector3i(192, 192, 192);
    COLORS[2]  = Eigen::Vector3i(255, 255, 0);
    COLORS[3]  = Eigen::Vector3i(255, 153, 18);
    COLORS[4]  = Eigen::Vector3i(255, 97,  0);
    COLORS[5]  = Eigen::Vector3i(176, 224, 230);
    COLORS[6]  = Eigen::Vector3i(65,  105, 225);
    COLORS[7]  = Eigen::Vector3i(0,   255, 255);
    COLORS[8]  = Eigen::Vector3i(0,   255, 0);
    COLORS[9]  = Eigen::Vector3i(64,  224, 208);
    COLORS[10] = Eigen::Vector3i(245, 222, 179);
    COLORS[11] = Eigen::Vector3i(255, 125, 64);
    COLORS[12] = Eigen::Vector3i(124, 252, 0);
    COLORS[13] = Eigen::Vector3i(160, 32,  240);
    COLORS[14] = Eigen::Vector3i(255, 0,   0);
    COLORS[15] = Eigen::Vector3i(255, 0,   255);
    COLORS[16] = Eigen::Vector3i(0,   0,   255);
    COLORS[17] = Eigen::Vector3i(221, 160, 221);
    COLORS[18] = Eigen::Vector3i(0,   199, 140);
    COLORS[19] = Eigen::Vector3i(255, 192, 203);
}


__inline CL_IO::~CL_IO()
{

}

#include "cl_io.cpp"
#endif // CL_IO_H

