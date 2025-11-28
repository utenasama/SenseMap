#include "utils.h"

#include <unordered_map>

namespace sensemap {

void ReadRigList(const std::string file_path, std::vector<RigNames>& list) {
    std::vector<sensemap::PlyPoint > points;
    if (!ExistsFile(file_path)){
        std::cout << "file is empty, " << file_path << std::endl;
        return;
    }
    std::ifstream ifs;
    //打开文件
    ifs.open(file_path.c_str(), std::ios::in);
    //定义一个字符串
    std::string str;
    //从文件中读取数据
    while(getline(ifs, str))
    {
        // std::cout << str << std::endl;
        std::string info_str = str;
        std::vector<std::string> strparams = StringSplit(info_str, ",");

        double time1 = std::atof(strparams.at(0).c_str());
        Eigen::Vector3d t1(std::atof(strparams.at(1).c_str()),
                           std::atof(strparams.at(2).c_str()),
                           std::atof(strparams.at(3).c_str()));

        Eigen::Vector4d q1(std::atof(strparams.at(7).c_str()),
                           std::atof(strparams.at(4).c_str()),
                           std::atof(strparams.at(5).c_str()),
                           std::atof(strparams.at(6).c_str()));

        std::string pcd;
        getline(ifs, pcd);
        std::string img1,img2,img3;
        getline(ifs, img1);
        getline(ifs, img2);
        getline(ifs, img3);
        
        std::string name = GetPathBaseName(pcd);
        name.substr(0, name.length() - 4);
        RigNames rig_name;
        rig_name.Init(name, time1, t1, q1,
                      "points/" + GetPathBaseName(pcd), 
                      "camera/front/" + GetPathBaseName(img1), 
                      "camera/left/" + GetPathBaseName(img2), 
                      "camera/right/" + GetPathBaseName(img3));

        list.push_back(rig_name);

        const auto R = QuaternionToRotationMatrix(q1);
        sensemap::PlyPoint pnt;
        pnt.x = t1.x();
        pnt.y = t1.y();
        pnt.z = t1.z();
        pnt.nx = R(0,0);
        pnt.ny = R(1,0);
        pnt.nz = R(2,0);
        points.push_back(pnt);
    }
    std::sort(list.begin(),list.end(),[](const RigNames& a ,const RigNames& b){
        return a.time < b.time;
    });
    std::cout << "read in " << list.size() << " frame." << std::endl;
    WriteTextPlyPoints(file_path + "_t.ply", points, true, false);
}

}
