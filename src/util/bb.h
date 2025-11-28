//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_BB_H_
#define SENSEMAP_UTIL_BB_H_

#include <iostream>
#include <vector>
#include <Eigen/Core>

namespace sensemap {

class BoundingBox {
public:
	BoundingBox() {
		lt[0] = lt[1] = lt[2] = 0.0f;
		rb[0] = rb[1] = rb[2] = 0.0f;
	}
	BoundingBox(float lt_[3],
				float rb_[3]) {
		lt = Eigen::Vector3f(lt_);
		rb = Eigen::Vector3f(rb_);
	}
	BoundingBox(const Eigen::Vector3f &lt_,
				const Eigen::Vector3f &rb_) {
		lt = Eigen::Vector3f(lt_);
		rb = Eigen::Vector3f(rb_);
	}

public:
	//              
	//              4_____________ 5
	//              /|           /|
	//             / |          / |
	//           7/__|_________/6 |
	//            |  |         |  |
	//            |	 |         |  |	
	//            |	0|_ _ _ _ _| _|1
	//            |	 /         |  /
	//            |	/          | /
	//            |/___________|/
	//            3            2
	//
	BoundingBox Get(int i) const {	
		Eigen::Vector3f corners[8];
		corners[0] = lt; corners[6] = rb;
		corners[1][0] = rb[0]; corners[1][1] = lt[1]; corners[1][2] = lt[2];
		corners[2][0] = rb[0]; corners[2][1] = rb[1]; corners[2][2] = lt[2];
		corners[3][0] = lt[0]; corners[3][1] = rb[1]; corners[3][2] = lt[2];
		corners[4][0] = lt[0]; corners[4][1] = lt[1]; corners[4][2] = rb[2];
		corners[5][0] = rb[0]; corners[5][1] = lt[1]; corners[5][2] = rb[2];
		corners[7][0] = lt[0]; corners[7][1] = rb[1]; corners[7][2] = rb[2];

		BoundingBox *bb = nullptr;
		switch (i) {
		case 0:
			bb = new BoundingBox(corners[0], (corners[0] + corners[6]) * 0.5f);
			break;
		case 1:
			bb = new BoundingBox((corners[0] + corners[1]) * 0.5f, (corners[1] + corners[6]) * 0.5f);
			break;
		case 2:
			bb = new BoundingBox((corners[0] + corners[2]) * 0.5f, (corners[2] + corners[6]) * 0.5f);
			break;
		case 3:
			bb = new BoundingBox((corners[0] + corners[3]) * 0.5f, (corners[3] + corners[6]) * 0.5f);
			break;
		case 4:
			bb = new BoundingBox((corners[0] + corners[4]) * 0.5f, (corners[4] + corners[6]) * 0.5f);
			break;
		case 5:
			bb = new BoundingBox((corners[0] + corners[5]) * 0.5f, (corners[5] + corners[6]) * 0.5f);
			break;
		case 6:
			bb = new BoundingBox((corners[0] + corners[6]) * 0.5f, corners[6]);
			break;
		case 7:
			bb = new BoundingBox((corners[0] + corners[7]) * 0.5f, (corners[7] + corners[6]) * 0.5f);
			break;
		default:
			bb = new BoundingBox;
			break;
		}
		return *bb;
	}
	const Eigen::Vector3f GetCorners(int i) const {
		Eigen::Vector3f corners[8];
		corners[0] = lt; corners[6] = rb;
		corners[1][0] = rb[0]; corners[1][1] = lt[1]; corners[1][2] = lt[2];
		corners[2][0] = rb[0]; corners[2][1] = rb[1]; corners[2][2] = lt[2];
		corners[3][0] = lt[0]; corners[3][1] = rb[1]; corners[3][2] = lt[2];
		corners[4][0] = lt[0]; corners[4][1] = lt[1]; corners[4][2] = rb[2];
		corners[5][0] = rb[0]; corners[5][1] = lt[1]; corners[5][2] = rb[2];
		corners[7][0] = lt[0]; corners[7][1] = rb[1]; corners[7][2] = rb[2];
		return corners[i];
	}
	Eigen::Vector3f GetCenter() const {
		return (lt + rb) * 0.5f;
	}

	const float GetMinLength() const {
		float lx = std::fabs(rb[0] - lt[0]);
		float ly = std::fabs(rb[1] - lt[1]);
		float lz = std::fabs(rb[2] - lt[2]);
		return std::min(lx, std::min(ly, lz));
	}

	const float GetMaxLength() const {
		float lx = std::fabs(rb[0] - lt[0]);
		float ly = std::fabs(rb[1] - lt[1]);
		float lz = std::fabs(rb[2] - lt[2]);
		return std::max(lx, std::max(ly, lz));
	}

	const bool Contains(const Eigen::Vector3f &p) const {
		float min_x = std::min(lt[0], rb[0]);
		float min_y = std::min(lt[1], rb[1]);
		float min_z = std::min(lt[2], rb[2]);
		float max_x = std::max(lt[0], rb[0]);
		float max_y = std::max(lt[1], rb[1]);
		float max_z = std::max(lt[2], rb[2]);
		return (p[0] >= min_x && p[0] <= max_x) &&
				(p[1] >= min_y && p[1] <= max_y) &&
				(p[2] >= min_z && p[2] <= max_z);
	}

public:
	Eigen::Vector3f lt, rb;
};

static void GetBoundingBox3D(const std::vector<Eigen::Vector3f> &points,
							 Eigen::Vector3f &lt, Eigen::Vector3f &rb) {

	lt[0] = lt[1] = lt[2] = std::numeric_limits<float>::max();
	rb[0] = rb[1] = rb[2] = std::numeric_limits<float>::lowest();

	int size = static_cast<int>(points.size());
	for (std::size_t i = 0; i < size; ++i) {
		lt[0] = std::min(lt[0], points[i][0]);
		lt[1] = std::min(lt[1], points[i][1]);
		lt[2] = std::min(lt[2], points[i][2]);

		rb[0] = std::max(rb[0], points[i][0]);
		rb[1] = std::max(rb[1], points[i][1]);
		rb[2] = std::max(rb[2], points[i][2]);
	}

	// std::cout << "min: " << lt.transpose() << std::endl;
	// std::cout << "max: " << rb.transpose() << std::endl;

	Eigen::Vector3f c = (lt + rb) * 0.5f;
	lt = (lt - c) * 1.0f + c;
	rb = (rb - c) * 1.0f + c;
}

} // namespace sensemap

#endif