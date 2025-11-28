//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <string>

#include <Eigen/Core>

namespace sensemap {

namespace graph {
void ExportToGraph(const std::string& filename,
                   const Eigen::MatrixXi& data);
} // namespace graph

} // namespace sensemap