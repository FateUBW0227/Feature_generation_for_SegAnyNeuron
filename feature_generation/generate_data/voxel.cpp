#include "voxel.h"
#include <fstream>
#include <tiffio.h>

bool Voxel_L::check(int x, int y, int z) {
    if (x < 0 || x >= dim_x || y < 0 || y >= dim_y || z < 0 || z >= dim_z) {
        return false;
    }
    return true;
}

void Voxel_L::malloc_memory(int val) {
    if (dim_x <= 0 || dim_y <= 0 || dim_z <= 0)
        std::cout << "error dimension. " << std::endl;
    data_ptr.clear();
    for (int i = 0; i < dim_z; ++i) {
        unsigned short *temp = new unsigned short[dim_x * dim_y];
        data_ptr.push_back(temp);

        for (int j = 0; j < dim_x; ++j)
            for (int k = 0; k < dim_y; ++k)
                temp[j + k * dim_x] = val;
    }
}

void Voxel_L::set_shape(int x, int y, int z, int val) {
    set_dim(x, y, z);
    malloc_memory(val);
}

void Voxel_L::write_image(std::string name) {
    TIFF *out            = TIFFOpen(name.c_str(), "w");
    uint16 bitspersample = 16, photo = 1, samplesperpixel = 1;

    unsigned char *buf = NULL;
    buf                = (unsigned char *)_TIFFmalloc(dim_x * sizeof(unsigned char));

    for (int depth = 0; depth < dim_z; ++depth) {
        if (depth % 100 == 0)
            std::cout << depth << std::endl;
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out, TIFFTAG_PAGENUMBER, dim_z);

        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bitspersample);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, photo);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesperpixel);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, 1);
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, dim_x);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, dim_y);

        TIFFSetDirectory(out, depth);
        for (int row = 0; row < dim_y; ++row) {
            // for (int col = 0; col < dim_x; ++col)
            //{
            //	buf[col] = (unsigned char)( this->operator()(col, row, depth) /*voxel(col, row, depth)*/);
            //}
            // TIFFWriteScanline(out, buf, row, 0);
            TIFFWriteScanline(out, data_ptr[depth] + row * dim_x, row, 0);
        }
        TIFFWriteDirectory(out);
    }
    TIFFClose(out);
    if (buf)
        _TIFFfree(buf);
    std::cout << "DONE " << std::endl;
}

void Voxel_L::write_image2(std::string name) {
    TIFF *out = TIFFOpen(name.c_str(), "w");
    for (int nz = 0; nz < dim_z; ++nz) {
        // We need to set some values for basic tags before we can add any data
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out, TIFFTAG_PAGENUMBER, 1);
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, dim_x);                   //图像的宽度
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, dim_y);                  //图像长度
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // 图像方向设置
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);                   //根据图像位深填不同的值
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);                  //每像素样本数
        TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW);        //压缩算法
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK); // 灰度映射
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);   // 单图像平面存储
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, dim_y);                 //每个条的行数
        TIFFSetField(out, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);        //图像排列方式
        TIFFWriteEncodedStrip(out, 0, data_ptr[nz], dim_x * dim_y);     //写文件
        TIFFWriteDirectory(out);
    }
    TIFFClose(out); //关闭文件
}

void Voxel_L::write_compressed_data(std::string file_name) {
    TIFF *out = TIFFOpen(file_name.c_str(), "w");
    for (int nz = 0; nz < dim_z; ++nz) {
        // We need to set some values for basic tags before we can add any data
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out, TIFFTAG_PAGENUMBER, 1);
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, dim_x);                   //图像的宽度
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, dim_y);                  //图像长度
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // 图像方向设置
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);                   //根据图像位深填不同的值
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);                  //每像素样本数
        TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW);        //压缩算法 
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK); // 灰度映射
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);   // 单图像平面存储
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, dim_y);                 //每个条的行数
        TIFFSetField(out, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);        //图像排列方式
        TIFFWriteEncodedStrip(out, 0, data_ptr[nz], dim_x * dim_y * 2);     //写文件
        TIFFWriteDirectory(out);
    }
    TIFFClose(out); //关闭文件
}

void Voxel_L::write_pts(std::string name) {
    std::ofstream fout(name.c_str());
    int idx = 1;
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < dim_y; ++j) {
            for (int k = 0; k < dim_z; ++k) {
                if (this->operator()(i, j, k) > 0) {
                    if (idx == 1)
                        fout << idx++ << " 0 " << i << " " << j << " " << k << " 0 -1 " << std::endl;
                    else
                        fout << idx++ << " 0 " << i << " " << j << " " << k << " 0 " << idx - 2 << std::endl;
                }
            }
        }
    }
    fout.close();
}

Voxel_L::Voxel_L(int x, int y, int z, int val) : dim_x(x), dim_y(y), dim_z(z) { malloc_memory(val); }

Voxel_L::~Voxel_L() {
    for (int i = 0; i < dim_z; ++i) {
        delete[] data_ptr[i];
    }
}

Voxel_E::Voxel_E(int x, int y, int z, int val) : dim_x(x), dim_y(y), dim_z(z) { malloc_memory(val); }

void Voxel_E::malloc_memory(int val) {
    if (dim_x <= 0 || dim_y <= 0 || dim_z <= 0)
        std::cout << "error dimension " << std::endl;
    data_ptr.clear();
    for (int i = 0; i < dim_z; ++i) {
        unsigned char *temp = new unsigned char[dim_x * dim_y];
        data_ptr.push_back(temp);

        for (int j = 0; j < dim_x; ++j) {
            for (int k = 0; k < dim_y; ++k)
                temp[j + k * dim_x] = val;
        }
    }
}

void Voxel_E::set_shape(int x, int y, int z, int val) {
    dim_x = x;
    dim_y = y;
    dim_z = z;
    malloc_memory(val);
}

void Voxel_E::write_compressed_data(std::string file_name) {
    TIFF *out = TIFFOpen(file_name.c_str(), "w");
    for (int nz = 0; nz < dim_z; ++nz) {
        // We need to set some values for basic tags before we can add any data
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out, TIFFTAG_PAGENUMBER, 1);
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, dim_x);                   //图像的宽度
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, dim_y);                  //图像长度
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // 图像方向设置
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);                    //根据图像位深填不同的值
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);                  //每像素样本数
        TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW);        //压缩算法
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK); // 灰度映射
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);   // 单图像平面存储
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, dim_y);                 //每个条的行数
        TIFFSetField(out, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);        //图像排列方式
        TIFFWriteEncodedStrip(out, 0, data_ptr[nz], dim_x * dim_y);     //写文件
        TIFFWriteDirectory(out);
    }
    TIFFClose(out); //关闭文件
}

void Voxel_E::write_image(std::string name) {
    TIFF *out            = TIFFOpen(name.c_str(), "w");
    uint16 bitspersample = 8, photo = 1, samplesperpixel = 1;

    unsigned char *buf = NULL;
    buf                = (unsigned char *)_TIFFmalloc(dim_x * sizeof(unsigned char));

    for (int depth = 0; depth < dim_z; ++depth) {
        if (depth % 100 == 0)
            std::cout << depth << std::endl;
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out, TIFFTAG_PAGENUMBER, dim_z);

        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bitspersample);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, photo);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesperpixel);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, 1);
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, dim_x);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, dim_y);

        TIFFSetDirectory(out, depth);
        for (int row = 0; row < dim_y; ++row) {
            // for (int col = 0; col < dim_x; ++col)
            //{
            //	buf[col] = (unsigned char)( this->operator()(col, row, depth) /*voxel(col, row, depth)*/);
            //}
            // TIFFWriteScanline(out, buf, row, 0);
            TIFFWriteScanline(out, data_ptr[depth] + row * dim_x, row, 0);
        }
        TIFFWriteDirectory(out);
    }
    TIFFClose(out);
    if (buf)
        _TIFFfree(buf);
    std::cout << "DONE " << std::endl;
}


Voxel_E::~Voxel_E() {
    for (int i = 0; i < dim_z; ++i) {
        delete[] data_ptr[i];
    }
}
