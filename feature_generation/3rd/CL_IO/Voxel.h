#ifndef VOLUMES_H
#define VOLUMES_H

#include <stddef.h>
#include <memory>
#include <QtCore/QDir>
#include <iostream>


template <typename T> class Voxel;
using IVoxel = Voxel<int>;
using DVoxel = Voxel<double>;


template <typename T>
class Voxel
{
public:
    Voxel();

    bool IsEmpty() const {return data_ptr == NULL;};

    T& operator() (size_t x, size_t y, size_t z) { return data_ptr[x + y * shape[0] + z * xy_]; };

    T operator() (size_t x, size_t y, size_t z) const { return data_ptr[x + y * shape[0] + z * xy_]; };

    bool Set_shape(const int &x, const int &y, const int &z);

    ~Voxel();

    size_t x() const {return shape[0];}
    size_t y() const {return shape[1];}
    size_t z() const {return shape[2];}
    int xy_;
//    friend class Gausscluster;

private:
    size_t shape[3] = {0, 0, 0};

    T* data_ptr = nullptr;
};


template<class T>
Voxel<T>::Voxel()
{

}


template<class T>
Voxel<T>::~Voxel()
{
	if (data_ptr != nullptr)
		delete[] data_ptr;
}


template <class T>
bool Voxel<T>::Set_shape(const int &x, const int &y, const int &z)
{
    shape[0] = x;
    shape[1] = y;
    shape[2] = z;
	std::cout << x << " " << y << " " << z << std::endl;
    xy_ = x * y;
    data_ptr = new T[x * y * z];
    for(int i = 0; i < x * y * z; i++)
        data_ptr[i] = 0;
    return true;
}


#include "Voxel.cpp"
#endif // VOLUME_H

