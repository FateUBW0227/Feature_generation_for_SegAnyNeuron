#include "cl_io.h"
#include <iomanip>


template bool CL_IO::load_swc<int>(const std::string name, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>
    &pts, Eigen::MatrixXd &others, bool block);

template bool CL_IO::load_swc<double>(const std::string name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
    &pts, Eigen::MatrixXd &others, bool block);

template bool CL_IO::load_mat<int> (const std::string name, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &mat,
    int cols, bool block);

template bool CL_IO::load_mat<double> (const std::string name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
    int cols, bool block);

template void CL_IO::COUT_VEC<double> (std::vector<double> array, std::string st);

template void CL_IO::COUT_VEC<int> (std::vector<int> array, std::string st);

__inline bool CL_IO::build_swc_structure(Eigen::MatrixXd &swc_data, std::vector<Eigen::MatrixXd> &structured_swc)
{
	if (swc_data.rows() != 7)
		return false;

	int pt_num = swc_data.cols();
	std::map<int, Eigen::VectorXd> pts_info;
	std::vector<int> idxs(pt_num);
	for (int i = 0; i < pt_num; ++i)
	{
		Eigen::VectorXd swc_data_new = swc_data.col(i);
		swc_data_new(1) = -1;
		pts_info.insert(std::make_pair(swc_data(0, i), swc_data_new));
		idxs[i] = swc_data(0, i);
	}

	int tree_idxs = 0;
	int iter_time = 1;
	do {
		std::vector<int> new_idxs;
		for (int i = 0; i < idxs.size(); ++i)
		{
			Eigen::VectorXd &tree_id = pts_info.at(idxs[i]);
			int head_id = tree_id(6);
			if (head_id == -1)
			{
				tree_id(1) = tree_idxs++;
			}
			else 
			{
				Eigen::VectorXd &head = pts_info.at(head_id);
				if (head(1) != -1)
				{
					tree_id(1) = head(1);
				}
				else
					new_idxs.push_back(idxs[i]);
			}
		}
		idxs = new_idxs;
	} while (idxs.size() > 0 && iter_time++ < (pt_num + 1));

	if (idxs.size() > 0)
		return false;

	std::vector<std::vector<double> > pts(tree_idxs);
	for(auto &item : pts_info)
	{
		int tree_id = item.second(1);
		for(int i = 0; i < 7; ++i)
			pts[tree_id].push_back(item.second(i));
	}

	structured_swc.resize(tree_idxs);
	for (int i = 0; i < tree_idxs; ++i)
	{
		structured_swc[i] = Eigen::Map<Eigen::MatrixXd>(pts[i].data(), 7, pts[i].size() / 7);
	}

	return true;
}


__inline int CL_IO::check_file_num(std::string suffix, std::string target_dir)
{
	QDir *dir = new QDir(QString(target_dir.c_str() ) );
	QStringList filter;
	filter <<  QString( suffix.c_str() );
	dir->setNameFilters(filter);
	QList<QFileInfo> *fileinfo = new QList<QFileInfo>(dir->entryInfoList(filter));
	int num = fileinfo->count();
	delete fileinfo;
	delete dir;
	return num;
}


__inline bool CL_IO::copy_file(std::string src_file, std::string tar_file)
{
	QFile::copy(QString::fromStdString(src_file), QString::fromStdString(tar_file) );

	return true;
}

__inline void CL_IO::COUT(std::string str)
{
    std::cout << str << std::endl;
}


__inline double CL_IO::Cout_Time(std::vector<std::chrono::steady_clock::time_point> &markers)
{
	markers.push_back(std::chrono::steady_clock::now());
	if (markers.size() < 2)
	{
		std::cout << "time used failed." << std::endl;
		exit(0);
	}
	return std::chrono::duration_cast<std::chrono::milliseconds>(markers[markers.size() - 1] - markers[markers.size() - 2]).count();
}


template <typename T>
__inline void CL_IO::COUT_VEC(std::vector<T> array, std::string st)
{
    std::cout << st;
    for(unsigned int i = 0; i < array.size(); ++i)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}


template <typename T>
__inline void CL_IO::COUT_VEC(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> array, std::string st)
{
    std::cout << st;
    if(array.rows() == 1)
    {
        for(unsigned int i = 0; i < array.cols(); ++i)
        {
            std::cout << array(0, i) << " ";
        }
    }
    if(array.cols() == 1)
    {
        for(unsigned int i = 0; i < array.rows(); ++i)
        {
            std::cout << array(i, 0) << " ";
        }
    }
    std::cout << std::endl;
}


__inline std::string CL_IO::dir(std::string dir_name)
{
	for (int i = 0; i < dir_name.size(); ++i)
	{
		if (dir_name[i] == '\\')
		{
			dir_name[i] = '/';
		}
	}
	if (dir_name[dir_name.size() - 1] != '/')
		dir_name += '/';
	return dir_name;
}


__inline double CL_IO::file_size(std::string name)
{
	QFileInfo info(QString(name.c_str()) );
	return info.size() / 1024.0;
}


template <typename T>
__inline bool CL_IO::load_mat(const std::string name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat,
                     int cols, bool block)
{
    std::ifstream fin(name);
    if(!fin.is_open() )
    {
        if(!block)
            std::cout << name << " not opened." << std::endl;
        return false;
    }
    std::vector<T> mat_v;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> __mat;
    T a;
    while(fin >> a)
    {
        mat_v.push_back(a);
    }
    if(mat_v.size() % cols != 0)
    {
        return false;
    }
    __mat.resize(cols, mat_v.size() / cols);
    memcpy(__mat.data(), mat_v.data(), sizeof(T) * mat_v.size());
    mat = __mat.transpose();

    return true;
}


template <typename T>
__inline bool CL_IO::load_swc(const std::string name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
              &pts, Eigen::MatrixXd &others, bool block)
{
    std::ifstream fin(name);
    if(!fin.is_open())
    {
        if(!block)
            std::cout << name << " not opened." << std::endl;
        return false;
    }
    std::vector<T> pts_v;
    std::vector<double> others_v;
    double a, b, c, d, e, f, g;
    while(fin >> a)
    {
        others_v.push_back(a);
        if(fin >> b)
            others_v.push_back(b);
        else
            return false;

        if(fin >> c)
             pts_v.push_back(static_cast<T>(c));
        else
            return false;

        if(fin >> d)
             pts_v.push_back(static_cast<T>(d));
        else
            return false;

        if(fin >> e)
             pts_v.push_back(static_cast<T>(e));
        else
            return false;

        if(fin >> f)
             others_v.push_back(f);
        else
            return false;

        if(fin >> g)
             others_v.push_back(g);
        else
            return false;
    }

    pts.resize(3, pts_v.size() / 3);
    memcpy(pts.data(), pts_v.data(), sizeof(T) * pts_v.size());

    others.resize(4, others_v.size() / 4);
    memcpy(others.data(), others_v.data(), sizeof(double) * others_v.size());

    return true;
}


template <typename T>
__inline bool CL_IO::load_swc(const std::string name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &swc_data)
{
	std::ifstream fin(name);
	if (!fin.is_open())
	{
		std::cout << name << "open failed." << std::endl;
		return false;
	}
	T id = 0, fiber_id = 0, x = 0, y(0), z(0), radius(0), head_id(0);
	std::string str;
	std::vector<T> swc_vec;
	while (std::getline(fin, str))
	{
		if (str[0] == '#')
			continue;
		std::stringstream ss(str);
		if ((ss >> id >> fiber_id >> x >> y >> z >> radius >> head_id))
		{
			swc_vec.push_back(id);
			swc_vec.push_back(fiber_id);
			swc_vec.push_back(x);
			swc_vec.push_back(y);
			swc_vec.push_back(z);
			swc_vec.push_back(radius);
			swc_vec.push_back(head_id);
		}
	}

	swc_data = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(swc_vec.data(), 7, swc_vec.size() / 7);
	return true;
}


__inline bool CL_IO::make_dirs(std::string name)
{
    QDir a;
	if (!a.exists(QString::fromStdString(name)))
	{
		if (!a.mkdir(QString::fromStdString( name ) ) )
		{
			std::cout << "\033[0;1;31m" << name << " create faied. " << "\033[0m" << std::endl;
		}
	
	}
	return true;
}


__inline std::vector<Eigen::MatrixXd> CL_IO::split_into_trees(Eigen::MatrixXd &swc_data)
{
	std::map<int, Eigen::VectorXd> ori_swc_infos;
	std::vector<int> idxs;
	for (int i = 0; i < swc_data.cols(); ++i)
	{
		int id = swc_data(0, i);
		swc_data(1, i) = -1;
		ori_swc_infos.insert(std::make_pair(id, swc_data.col(i)));
		idxs.push_back(id);
	}

	int tree_id = 0;
	int iter_time = 0;
	do {
		std::vector<int> un_idxs;
		for (int i = 0; i < idxs.size(); ++i)
		{
			if (ori_swc_infos.at(idxs[i])(1) == -1)
			{
				int head_id = ori_swc_infos.at(idxs[i])(6);
				if (head_id == -1)
				{
					ori_swc_infos.at(idxs[i])(1) = tree_id;
					tree_id++;
				}
				else if (ori_swc_infos.at(head_id)(1) != -1)
				{
					ori_swc_infos.at(idxs[i])(1) = ori_swc_infos.at(head_id)(1);
				}
				else
				{
					un_idxs.push_back(idxs[i]);
				}
			}
		}
		iter_time++;
		idxs = un_idxs;
	} while (idxs.size() > 0 && iter_time < swc_data.cols());

	if (idxs.size() > 0)
	{
		std::cout << "check input." << std::endl;
		exit(0);
	}

	std::vector<std::vector<Eigen::VectorXd> > trees(tree_id);
	for (auto &info : ori_swc_infos)
	{
		int tree_id = info.second(1);
		trees[tree_id].push_back(info.second);
	}

	std::vector<Eigen::MatrixXd> new_swc_data(tree_id);
	for (int i = 0; i < tree_id; ++i)
	{
		new_swc_data[i].resize(7, trees[i].size());
		for (int j = 0; j < trees[i].size(); ++j)
			new_swc_data[i].col(j) = trees[i][j];
	}
	return new_swc_data;
}


__inline bool CL_IO::re_scale_swc(std::string src_file, std::string tar_file, double ratio0, double ratio1, double ratio2)
{
	Eigen::MatrixXd swc_data;
	CL_IO::load_swc(src_file, swc_data);
	for (int i = 0; i < swc_data.cols(); ++i)
	{
		swc_data(2, i) *= ratio0;
		swc_data(3, i) *= ratio1;
		swc_data(4, i) *= ratio2;
	}
	CL_IO::save_swc(tar_file, swc_data);
	return true;
}


__inline std::vector<std::string> CL_IO::split(std::string str, char character)
{
	std::vector<std::string> split_str;
	int start_id = 0;
	for (int i = 0; i < str.size(); ++i)
	{
		if (str[i] == character)
		{
			split_str.push_back(str.substr(start_id, i - start_id));
			start_id = i + 1;
		}
	}
	split_str.push_back(str.substr(start_id, str.size() - start_id));
	return split_str;
}

__inline bool CL_IO::write_image(std::string name, IVoxel &voxel)
{
    int x = static_cast<int>(voxel.x()), y = static_cast<int>(voxel.y()), z = static_cast<int>(voxel.z());
    TIFF *out = TIFFOpen(name.c_str(), "w");
    uint16 bitspersample = 16, photo = 1, samplesperpixel = 1;

    unsigned short * buf = NULL;
    buf = (unsigned short *)_TIFFmalloc(x * sizeof(unsigned short));

    for(int depth = 0; depth < z; ++depth)
    {
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out,TIFFTAG_PAGENUMBER, z);

        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bitspersample);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, photo);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesperpixel);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, 1);
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, x);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, y);

        TIFFSetDirectory(out, depth);
        for(int row = 0; row < y; ++row)
        {
            for(int col = 0; col < x; ++col)
            {
                buf[col] = (unsigned short)(voxel(col, row, depth));
            }
            TIFFWriteScanline(out, buf, row, 0);
        }
        TIFFWriteDirectory(out);
    }
    TIFFClose(out);
    if(buf)
        _TIFFfree(buf);
    std::cout << "DONE " << std::endl;
    return false;
}


__inline void CL_IO::TIME(int id)
{

}


__inline bool CL_IO::increase_density(Eigen::MatrixXd &swc_data)
{
	std::map<int, Eigen::VectorXd> swc_map;
	for (int i = 0; i < swc_data.cols(); ++i)
	{
		swc_map.insert(std::make_pair(swc_data(0, i), swc_data.col(i)) );
	}

	int current_id = 0;
	std::vector<double> swc_vec;
	int total_id = 1;
	for (auto &idx : swc_map)
	{
		if (idx.second(6) == -1)
		{
			swc_vec.push_back(total_id);
			idx.second(0) = total_id;
			for (int i = 0; i < 6; ++i)
			{
				swc_vec.push_back(idx.second(i + 1));
			}
			total_id += 1;
		}
		else
		{
			int head_id = idx.second(6);
			Eigen::VectorXd head_info = swc_map.at(head_id);
			Eigen::VectorXd& info = idx.second;
			double dist = (head_info.block(2, 0, 3, 1) - info.block(2, 0, 3, 1)).norm();
			//std::cout << head_info.transpose() << " / " << info.transpose() << std::endl;
			if (dist > 1.0)
			{
				int insert_num = dist + 1.0;
				for (int j = 0; j < insert_num; ++j)
				{
					Eigen::Vector3d insert_pt = double(j + 1) / insert_num * info.block(2, 0, 3, 1)
						+ (1 - double(j + 1) / insert_num) * head_info.block(2, 0, 3, 1);
					if (j == 0)
					{
						Eigen::VectorXd insert_swc(7);
						insert_swc(0) = total_id; insert_swc(1) = 1.0;
						insert_swc(2) = insert_pt(0); insert_swc(3) = insert_pt(1); insert_swc(4) = insert_pt(2);
						insert_swc(5) = 1.0; insert_swc(6) = head_info(0);
						for (int i = 0; i < 7; ++i)
							swc_vec.push_back(insert_swc(i));
						total_id++;
					}
					else
					{
						Eigen::VectorXd insert_swc(7);
						insert_swc(0) = total_id; insert_swc(1) = 1.0;
						insert_swc(2) = insert_pt(0); insert_swc(3) = insert_pt(1); insert_swc(4) = insert_pt(2);
						insert_swc(5) = 1.0; insert_swc(6) = total_id - 1;
						for (int i = 0; i < 7; ++i)
							swc_vec.push_back(insert_swc(i));
						if (j == insert_num - 1)
						{
							info(0) = total_id;
						}
						total_id ++;
					}
				}
			}
			else
			{
				swc_vec.push_back(total_id);
				idx.second(0) = total_id;
				for (int i = 0; i < 5; ++i)
				{
					swc_vec.push_back(idx.second(i + 1));
				}
				swc_vec.push_back(head_info(0));
				total_id += 1;
			}
		}
	}

	swc_data = Eigen::Map<Eigen::MatrixXd>(swc_vec.data(), 7, swc_vec.size() / 7);
	return true;
}


__inline bool CL_IO::read_image(std::string file_name, IVoxel &voxel, bool block)
{
    TIFF *in = TIFFOpen(file_name.c_str(), "r");
    if(!in)
    {
        return false;
    }

    uint16 bitspersample = 8;
    uint16 photo = 1;   //1 : 0 is dark, 255 is white
    uint16 samplesperpixel = 1;  //8bit denote a pixel
    TIFFGetField(in, TIFFTAG_BITSPERSAMPLE, &bitspersample);
    TIFFGetField(in, TIFFTAG_PHOTOMETRIC, &photo); // PHOTOMETRIC_MINISWHITE=0, PHOTOMETRIC_MINISBLACK=1
    TIFFGetField(in, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel);

    int _x, _y, _z;
    TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &_x);
    TIFFGetField(in, TIFFTAG_IMAGELENGTH, &_y);
    _z = TIFFNumberOfDirectories(in);
    voxel.Set_shape(_x, _y, _z);

    if (bitspersample == 8) {
        TIFFSetDirectory(in, 0);
        uint8* data = new uint8[_x];
        for(int nCur = 0; nCur < _z; ++nCur)
        {
            for (int j = 0; j < _y; ++j)
            {
                TIFFReadScanline(in, data, j, 0);
                for (int i = 0; i < _x; ++i)
                {
                    voxel(i, j, nCur) = data[i];
                }
            }
            TIFFReadDirectory(in); //read next page
        }
        delete[] data;
    }
    else
    {
        uint16 *data = new uint16[_x * _y];
        int width(0), height(0);
        TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(in, TIFFTAG_IMAGELENGTH, &height);
        memset(data, 0, sizeof(short)* width * height);
        TIFFSetDirectory(in, 0);
        int StripSize = TIFFStripSize(in);
        int NumberofStrips = TIFFNumberOfStrips(in);
        for (int nCur = 0; nCur < _z; ++nCur) {
            for (int i = 0; i < NumberofStrips; ++i) {
                TIFFReadEncodedStrip(in, i, (data + i * StripSize / sizeof(short)), StripSize);//
            }
                for (int i = 0; i < _x; ++i)
                {
                    for (int j = 0; j < _y; ++j)
                    {
                        voxel(i, j, nCur) = data[i + j * _x];
                    }
                }
            TIFFReadDirectory(in);//read next page
        }
        delete[] data;
    }
    TIFFClose(in);
    if(!block)
        std::cout << file_name << " bit " << bitspersample << std::endl;
    return true;
}


__inline bool CL_IO::obtain_list(std::string dir_name, std::vector<std::string> &file_name,
                        std::string suffix)
{
    file_name.clear();
    QStringList nameFilters;
    if(suffix != "")
    {
        nameFilters.push_back(QString::fromStdString(suffix));
    }
    QDir dir(QString::fromStdString(dir_name));
    QStringList list = dir.entryList(nameFilters, QDir::Files|QDir::Readable, QDir::Name);
    for(int i = 0; i < list.size(); i++)
    {
        file_name.push_back(list[i].toStdString());
    }
    if(file_name.size() == 0)
    {
        return false;
    }
    return true;
}


__inline bool CL_IO::output2matlab(double *data, std::string file_name, std::string variable_name,
                              int row, int col)
{
//    MATFile *pmatfile = NULL;
//    pmatfile = matOpen(file_name.c_str(), "w");
//    mxArray *pMxArray = NULL;
//    pMxArray = mxCreateDoubleMatrix(row, col, mxREAL);
//    mxSetData(pMxArray, data);
//    matPutVariable(pmatfile, variable_name.c_str(), pMxArray);
//    matClose(pmatfile);
    return true;
}


template <typename T>
__inline bool CL_IO::save_swc(std::string file_name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &swc_data)
{
	std::ofstream fout(file_name);
	if (!fout.is_open())
	{
		std::cout << file_name << " open failed. " << std::endl;
		return false;
	}
	if (swc_data.rows() != 7)
	{
		std::cout << file_name << " wrong row." << std::endl;
		return false;
	}

	fout << std::setprecision(7);
	for (int i = 0; i < swc_data.cols(); ++i)
	{
		fout << swc_data(0, i) << " " << swc_data(1, i) << " " << swc_data(2, i) << " " << swc_data(3, i) << " " << 
			    swc_data(4, i) << " " << swc_data(5, i) << " " << swc_data(6, i) << std::endl;
	}
	fout.close();
	return true;
}


__inline bool CL_IO::save_time(std::vector<std::chrono::steady_clock::time_point> &markers)
{
	markers.push_back(std::chrono::steady_clock::now());
	return true;
}
