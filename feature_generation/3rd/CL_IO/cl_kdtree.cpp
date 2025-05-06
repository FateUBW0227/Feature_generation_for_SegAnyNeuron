#include "cl_kdtree.h"


__inline KDTree::KDTree()
{

}


__inline KDTree::~KDTree()
{
	if (_dim != 0)
	{
		annDeallocPts(dataPts);
		delete kdtree;
	}
	annClose();
}


__inline bool KDTree::build_tree(std::vector<Eigen::Vector3d> pts, int dim)
{
    if(pts.size() == 0)
    {
        return false;
    }

    _dim = dim;
    dataPts = annAllocPts(static_cast<int>(pts.size()), dim); 
    for(int i = 0; i < pts.size(); ++i)
    {
        for(int j = 0; j < dim; j++)
        {
            dataPts[i][j] = pts[i](j);
        }
    }
	pt_num = pts.size();
    kdtree = new ANNkd_tree(dataPts, static_cast<int>(pts.size()), dim);
    return true;
}


__inline bool KDTree::build_tree(std::vector<Eigen::VectorXd> pts, int dim)
{
    if(pts.size() == 0)
    {
        return false;
    }

    if(dim != pts[0].rows())
    {
        std::cout << "dim not correspond to matrix in " << __LINE__ << " cl_kdtree.cpp" << std::endl;
        return false;
    }

    _dim = dim;
    dataPts = annAllocPts(static_cast<int>(pts.size()), dim); 
    for(int i = 0; i < pts.size(); ++i)
    {
        for(int j = 0; j < dim; j++)
        {
            dataPts[i][j] = pts[i](j);
        }
    }
	pt_num = pts.size();
    kdtree = new ANNkd_tree(dataPts, static_cast<int>(pts.size()), dim);
    return true;
}


__inline bool KDTree::build_tree(Eigen::MatrixXd pts, int dim)
{
    if(dim != pts.rows())
    {
        std::cout << "dim not correspond to matrix in " << __LINE__ << " cl_kdtree.cpp" << std::endl;
        return false;
    }
    //_dim = dim;
    dataPts = annAllocPts(static_cast<int>(pts.cols()), dim); // may leak memory
    for(int i = 0; i < pts.cols(); ++i)
    {
        for(int j = 0; j < dim; j++)
        {
            dataPts[i][j] = pts(j, i);
        }
    }
	_dim = dim;
	pt_num = pts.cols();
    kdtree = new ANNkd_tree(dataPts, static_cast<int>(pts.cols()), dim);
    return true;
}


__inline bool KDTree::search_pts(Eigen::VectorXd pt, int *idx, double *dists, int neigbor_num)
{
    if(pt.rows() != _dim)
    { 
        std::cout << "dim not correspond to matrix in line " << __LINE__ << " cl_kdtree.cpp" << std::endl;
        return false;
    }
    ANNpoint queryPt;
    queryPt = annAllocPt(_dim);
    for(int i = 0; i < _dim; i++)
    {
        queryPt[i] = pt(i);
    }
    kdtree->annkSearch(queryPt, neigbor_num, idx, dists, 0.0);
	annDeallocPt(queryPt);
    return true;
}


__inline bool KDTree::search_pts(Eigen::VectorXd pt, Res &res) // square distance
{
	if (pt.rows() != _dim)
	{
		std::cout << pt.rows() << " " << _dim << std::endl;
		std::cout << "dim not correspond to matrix in line " << __LINE__ << " cl_kdtree.cpp" << std::endl;
		return false;
	}
	ANNpoint queryPt;
	queryPt = annAllocPt(_dim);
	for (int i = 0; i < _dim; ++i)
		queryPt[i] = pt[i];
	kdtree->annkSearch(queryPt, res.num, res.idxs.data(), res.dists.data(), 0.0);
	return true;
}