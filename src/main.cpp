#include <nanoflann.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <iomanip>

static void loadXYZ(const std::string filename, std::vector<Eigen::Vector3d>& pts, std::vector<Eigen::Vector3d>& nor)
{
    std::ifstream file(filename.c_str(),std::ifstream::in);
    if( file.fail() == true )
    {
        std::cerr << filename << " could not be opened" << std::endl;
    }

    int i=0;
    while(file){
        Eigen::Vector3d pt,no;
        file >> pt.x() >> pt.y() >> pt.z() >> no.x() >> no.y() >> no.z();
        pts.push_back(pt);
        nor.push_back(no);
    }
}

static void writeXYZ(const std::string filename, const std::vector<Eigen::Vector3d>& pts, const std::vector<Eigen::Vector3d>& nor)
{
    std::ofstream file(filename.c_str(), std::ofstream::out);
    if (file.fail() == true)
    {
        std::cerr << filename << " could not be opened" << std::endl;
    }

    for (int i = 0; i < pts.size(); i++) {
        const Eigen::Vector3d& pt = pts[i];
        const Eigen::Vector3d& no = nor[i];
        file << std::fixed << std::setprecision(6) << 
            pt.x() << " " << pt.y() << " " << pt.z() << " " <<
            no.x() << " " << no.y() << " " << no.z() << " " << std::endl;
    }
}

static void pointSetPCA(const std::vector<Eigen::Vector3d>& pts, Eigen::Vector3d& normal) {

    assert(pts.size() >= 3); //otherwise normals are undetermined
    Eigen::Map<const Eigen::Matrix3Xd> P(&pts[0].x(), 3, pts.size());

    Eigen::Vector3d centroid = P.rowwise().mean();
    Eigen::MatrixXd centered = P.colwise() - centroid;
    Eigen::Matrix3d cov = centered * centered.transpose();

    //eigvecs sorted in increasing order of eigvals
    //TODO: move out of function and reuse to increase speed 
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;//(cov);
    //eig.compute(cov);
    eig.computeDirect(cov); //https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html#afe520161701f5f585bcc4cedb8657bd1
    normal = eig.eigenvectors().col(0); //is already normalized
    if (normal(2) > 0) normal = -normal; //flip towards camera
}

struct StdVecOfEigenVector3dRefAdapter
{
public:
    StdVecOfEigenVector3dRefAdapter(const std::vector<Eigen::Vector3d>& pps) : pts(pps) {};

    const std::vector<Eigen::Vector3d>& pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline double kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim == 0) return pts[idx].x();
        else if (dim == 1) return pts[idx].y();
        else return pts[idx].z();
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

std::vector<Eigen::Vector3d> computeNormals(const std::vector<Eigen::Vector3d>& pts, int num_results = 10) {

   std::vector<Eigen::Vector3d> nor(pts.size());

   // construct a kd-tree index:
   typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double,StdVecOfEigenVector3dRefAdapter>,StdVecOfEigenVector3dRefAdapter,3> my_kd_tree_t;

   StdVecOfEigenVector3dRefAdapter cloud(pts);

   my_kd_tree_t index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
   index.buildIndex();

   std::vector<size_t>  ret_index(num_results);
   std::vector<double> out_dist_sqr(num_results);

   size_t l = pts.size();

   for (int i = 0; i < l; i++) {
       int num_results_actual = index.knnSearch(pts[i].data(), num_results, &ret_index[0], &out_dist_sqr[0]);

       std::vector<Eigen::Vector3d> neighbours(num_results_actual+1);//TODO speedup by avoiding to copy indexed points to continuous pts mat
       neighbours[0] = pts[i]; //add point itself to pointset
       for (int j = 0; j < num_results_actual; j++) {
           neighbours[j+1] = pts[ret_index[j]];
       }

       pointSetPCA(neighbours, nor[i]);
   }

   return nor;
}

int main(){
    std::vector<Eigen::Vector3d> pts,nor;

    loadXYZ("../cloudXYZ_0.xyz",pts,nor);

    std::cout<< pts[0].transpose() << "  " << nor[0].transpose() <<std::endl;

    std::vector<Eigen::Vector3d> nor2 = computeNormals(pts);

    double pi = 3.1415926536;

    for (int i = 0; i < pts.size(); i += pts.size() / 10) {
        double angle = 180 * acos(nor[i].dot(nor2[i])) / pi;
        std::cout << "i:" << i << "\t diff=" << angle << std::endl;
        //" n: " << nor[i].transpose() << " n2: " << nor2[i].transpose() << std::endl;
    }


    writeXYZ("../cloudXYZ_0-out.xyz", pts, nor2);

}