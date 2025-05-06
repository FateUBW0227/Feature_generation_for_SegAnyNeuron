#ifndef VOXEL_H
#define VOXEL_H
#include <Eigen/Dense>
#include <iostream>
#include <vector>

// 16bit
class Voxel_L {
  public:
    bool check(int x, int y, int z);
    void malloc_memory(int val = 0);
    void set_dim(int x, int y, int z) {
        dim_x = x;
        dim_y = y;
        dim_z = z;
    };
    void set_shape(int x, int y, int z, int val = 0);
    void write_image(std::string name);
    void write_image2(std::string name);
    void write_pts(std::string name);
    void write_compressed_data(std::string file_name);

    unsigned short &operator()(size_t x, size_t y, size_t z) { return data_ptr[z][x + y * dim_x]; };

    Voxel_L(){};
    Voxel_L(int x, int y, int z, int val = 0);
    ~Voxel_L();
    std::vector<unsigned short *> data_ptr;
    int dim_x = 0, dim_y = 0, dim_z = 0;
};

// 8bit
class Voxel_E {

  public:
    Voxel_E(){};
    Voxel_E(int x, int y, int z, int val = 0);
    void malloc_memory(int val = 0);
    void set_shape(int x, int y, int z, int val = 0);
    void write_compressed_data(std::string file_name);
    void write_image(std::string name);
    ~Voxel_E();

    unsigned char &operator()(size_t x, size_t y, size_t z) { return data_ptr[z][x + y * dim_x]; };
    unsigned char &operator()(Eigen::Vector3d coord) { return operator()(coord(0), coord(1), coord(2)); };

    std::vector<unsigned char *> data_ptr;
    int dim_x = 0, dim_y = 0, dim_z = 0;
};

#endif // ! VOXEL_H
