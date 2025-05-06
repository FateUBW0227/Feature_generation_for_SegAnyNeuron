#ifndef CL_KDTREE_H
#define CL_KDTREE_H

#include <ANN/ANN.h>
#include <Eigen/Dense>
#include <vector>

struct Res
{
	Res(int n) :num(n)
	{
		dists.resize(num);
		idxs.resize(num);
	}

	void show_res()
	{
		for (int i = 0; i < num; ++i)
		{
			std::cout << "{ " << idxs(i) << ", " << dists(i) << "}" << std::endl;
		}
	}

	Eigen::VectorXd dists;
	Eigen::VectorXi idxs;
	int num;
};


class KDTree
{
public:
	KDTree();
    ~KDTree();
    ANNkd_tree *kdtree;
    bool build_tree(Eigen::MatrixXd pts, int dim);
    bool build_tree(std::vector<Eigen::VectorXd> pts, int dim);
    bool build_tree(std::vector<Eigen::Vector3d> pts, int dim = 3);
    bool search_pts(Eigen::VectorXd pt, int *idx, double *dists, int neigbor_num);
	bool search_pts(Eigen::VectorXd pt, Res &res);

	ANNpointArray dataPts;
	int pt_num = 0;
    int _dim = 0;
};


#ifndef kdtree_static
#include "cl_kdtree.cpp"
#endif
#endif // CL_KDTREE_H
