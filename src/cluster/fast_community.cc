////////////////////////////////////////////////////////////////////////
// --- COPYRIGHT NOTICE ---------------------------------------------
// FastCommunityMH - infers community structure of networks
// Copyright (C) 2004 Aaron Clauset
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// 
// See http://www.gnu.org/licenses/gpl.txt for more details.
// 
////////////////////////////////////////////////////////////////////////
// Author       : Aaron Clauset  (aaron@cs.unm.edu)				//
// Location     : U. Michigan, U. New Mexico						//
// Time         : January-August 2004							//
// Collaborators: Dr. Cris Moore (moore@cs.unm.edu)				//
//              : Dr. Mark Newman (mejn@umich.edu)				//
////////////////////////////////////////////////////////////////////////
// --- DEEPER DESCRIPTION ---------------------------------------------
//  see http://www.arxiv.org/abs/cond-mat/0408187 for more information
// 
//  - read network structure from data file (see below for constraints)
//  - builds dQ, H and a data structures
//  - runs new fast community structure inference algorithm
//  - records Q(t) function to file
//  - (optional) records community structure (at t==cutstep)
//  - (optional) records the list of members in each community (at t==cutstep)
//
////////////////////////////////////////////////////////////////////////
// --- PROGRAM USAGE NOTES ---------------------------------------------
// This program is rather complicated and requires a specific kind of input,
// so some notes on how to use it are in order. Mainly, the program requires
// a specific structure input file (.pairs) that has the following characteristics:
//  
//  1. .pairs is a list of tab-delimited pairs of numeric indices, e.g.,
//		"54\t91\n"
//  2. the network described is a SINGLE COMPONENT
//  3. there are NO SELF-LOOPS or MULTI-EDGES in the file; you can use
//     the 'netstats' utility to extract the giantcomponent (-gcomp.pairs)
//     and then use that file as input to this program
//  4. the MINIMUM NODE ID = 0 in the input file; the maximum can be
//     anything (the program will infer it from the input file)
// 
// Description of commandline arguments
// -f <filename>    give the target .pairs file to be processed
// -l <text>		the text label for this run; used to build output filenames
// -t <int>		timer period for reporting progress of file input to screen
// -s			calculate and record the support of the dQ matrix
// -v --v ---v		differing levels of screen output verbosity
// -o <directory>   directory for file output
// -c <int>		record the aglomerated network at step <int>
// 
////////////////////////////////////////////////////////////////////////
// Change Log:
// 2006-02-06: 1) modified readInputFile to be more descriptive of its actions
//             2) removed -o functionality; program writes output to directory
//             of input file. (also removed -h option of command line)
// 2006-10-13: 3) Janne Aukia (jaukia@cc.hut.fi) suggested changes to the 
//             mergeCommunities() function here (see comments in that function),
//             and an indexing adjustment in printHeapTop10() in maxheap.h.
//
////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <string.h>
#include "stdlib.h"
#include "time.h"
#include "math.h"

#include "maxheap.h"
#include "vektor.h"

#include "fast_community.h"

// CGAL: depth-map initialization
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Projection_traits_xy_3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <CGAL/tags.h>

#include <CGAL/Epick_d.h>
#include <CGAL/point_generators_d.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Fuzzy_iso_box.h>
#include <CGAL/Search_traits_d.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/Convex_hull_traits_adapter_2.h>
#include <CGAL/property_map.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/mutex.h>
#endif

#include "util/threading.h"

#define VERBOSE

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
typedef CGAL::Convex_hull_traits_adapter_2<K, CGAL::Pointer_property_map<Point_2>::type > Convex_hull_traits_2;

namespace fastcommunity {

using namespace sensemap;
using namespace std;

// ------------------------------------------------------------------------------------
// Edge object - defined by a pair of vertex indices and *edge pointer to next in linked-list
class edge {
public:
	int     so;					// originating node
	int     si;					// terminating node
	double  weight;
	edge    *next;					// pointer for linked list of edges
	
	edge();						// default constructor
	~edge();						// default destructor
};
edge::edge()  { so = 0; si = 0; next = NULL; }
edge::~edge() {}

// ------------------------------------------------------------------------------------
// Nodenub object - defined by a *node pointer and *node pointer 
struct nodenub {
	tuple	*heap_ptr;			// pointer to node(max,i,j) in max-heap of row maxes
	vektor    *v;					// pointer stored vector for (i,j)
};

// ordered pair structures (handy in the program)
struct apair { int x; int y; };

// ------------------------------------------------------------------------------------
// List object - simple linked list of integers
class list {
public:
	int		index;				// node index
	list		*next;				// pointer to next element in linked list
	list();   ~list();
};
list::list()  { index= 0; next = NULL; }
list::~list() {}

// ------------------------------------------------------------------------------------
// Community stub object - stub for a community list
class stub {
public:
	bool		valid;				// is this community valid?
	int		size;				// size of community
	list		*members;				// pointer to list of community members
	list		*last;				// pointer to end of list
	stub();   ~stub();
};
stub::stub()  { valid = false; size = 0; members = NULL; last = NULL; }
stub::~stub() {
	list *current;
	if (members != NULL) {
		current = members;
		while (current != NULL) { members = current->next; delete current; current = members; }
	}
}

// ------------------------------------------------------------------------------------
// PROGRAM PARAMETERS -----------------------------------------------------------------

struct netparameters {
	int			n;				// number of nodes in network
	int			m;				// number of edges in network
	int     w;        // sum weight of edges in network
	int			maxid;			// maximum node id
	int			minid;			// minimum node id
}; netparameters    gparm;

struct groupstats {
	int			numgroups;		// number of groups
	double		meansize;			// mean size of groups
	int			maxsize;			// size of largest group
	int			minsize;			// size of smallest group
	double		*sizehist;		// distribution of sizes
}; groupstats		gstats;

struct outparameters {
	short int		textFlag;			// 0: no console output
								// 1: writes file outputs
	bool			suppFlag;			// T: no support(t) file
								// F: yes support(t) file
	short int		fileFlag;			// 
	string		filename;			// name of input file
	string		d_in;			// (dir ) directory for input file
	string		d_out;			// (dir ) director for output file
	string		f_parm;			// (file) parameters output
	string		f_input;			// (file) input data file
	string		f_joins;			// (file) community hierarchy
	string		f_support;		// (file) dQ support as a function of time
	string		f_net;			// (file) .wpairs file for .cutstep network
	string		f_group;			// (file) .list of indices in communities at .cutstep
	string		f_gstats;			// (file) distribution of community sizes at .cutstep
	string		s_label;			// (temp) text label for run
	string		s_scratch;		// (temp) text for building filenames
	int			timer;			// timer for displaying progress reports 
	bool			timerFlag;		// flag for setting timer
	int			cutstep;			// step at which to record aglomerated network
}; outparameters	ioparm;

// ------------------------------------------------------------------------------------
// ----------------------------------- GLOBAL VARIABLES -------------------------------

char		pauseme;
edge		*e;				// initial adjacency matrix (sparse)
edge		*elist;			// list of edges for building adjacency matrix
nodenub   *dq;				// dQ matrix
maxheap   *h;				// heap of values from max_i{dQ_ij}
double    *Q;				// Q(t)
dpair     Qmax;			// maximum Q value and the corresponding time t
double    *a;				// A_i
apair	*joins;			// list of joins
stub		*c;				// link-lists for communities

enum {NONE};

int    supportTot;
double supportAve;

namespace {

typedef std::pair<uint32_t, uint32_t> IMAGE_PAIR;
typedef std::pair<int, int> PAIR_II;
typedef PAIR_II GRAPH_EDGE;

typedef struct Cluster {
	int cluster_id;
	std::vector<uint32_t> image_ids;
	uint8_t cross_merge = 0;
} Cluster;

int Find(int x, std::vector<int>& par);
void UnionSet(int x, int y, std::vector<int>& par, std::vector<int>& rank);
void BuildLabels(int t, std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options);
void MergeClusters(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	const std::unordered_map<uint32_t, Eigen::Vector3d> &image_pose,
	std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options);
void MergePieces(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options);
void AddOverlapImages(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	const std::vector<float> &weights,
	const std::unordered_map<uint32_t, std::unordered_set<uint32_t> >& neighbors,
	std::vector<Cluster*> &clusters,
	std::vector<std::unordered_set<uint32_t> > &overlaps,
	const sensemap::SceneClustering::Options& options);

std::list<CommunityTree<uint32_t> *> MergeCommunityTree(
    CommunityTree<uint32_t> *root,
	const sensemap::SceneClustering::Options& options);
void FastBuildCommunityTree(
		const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
		CommunityTree<uint32_t> *community_tree,
	const sensemap::SceneClustering::Options& options);
void FastCommunityRecurisive(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options);
void FastCommunity(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options);
void BuildAdjacentMatrix(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids);
void buildDeltaQMatrix();
void buildFilenames();
void dqSupport();
void groupListsSetup();
void groupListsStats();
void groupListsUpdate(const int x, const int y);
void mergeCommunities(int i, int j);
bool parseCommandLine(int argc,char * argv[]);
void readInputFile();
void recordGroupLists();
void recordNetwork();

}

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3                                     Point_3;
typedef boost::tuple<Point_3,int>                           Point_and_int;
typedef CGAL::Search_traits_3<Kernel>                       Traits_base;
typedef CGAL::Search_traits_adapter<Point_and_int,
CGAL::Nth_of_tuple_property_map<0, Point_and_int>,
Traits_base>                                              Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>          K_neighbor_search;
typedef K_neighbor_search::Tree                             Tree;
typedef K_neighbor_search::Distance                         Distance;
typedef K_neighbor_search::Point_with_transformed_distance  Point_with_distance;

void ConvertCorrDist2Pair(
	const std::shared_ptr<sensemap::SceneGraphContainer>& scene_graph_container,
	const std::set<uint32_t>& valid_image_ids,
	std::vector<std::tuple<uint32_t, uint32_t, double> >& image_pairs){
	
	std::unordered_map<uint64_t, double> dist_between_images;
	int nb_neighbors = 50;

  	const size_t num_images = scene_graph_container->NumImages();
	const auto& images = scene_graph_container->Images();

	std::vector<Point_3> points(num_images);
  	std::vector<size_t> indices(num_images);
	std::size_t i = 0;
	for (const auto& image : images) {
		Eigen::Vector3d tvec = image.second.TvecPrior();
		points[i] = Point_3(tvec.x(), tvec.y(), tvec.z());
		indices[i] = image.first;
		// std::cout << "tuple: " << i << "-" << indices[i] << ", " << tvec.transpose() << std::endl;
		i++;
	}

	// Instantiate a KD-tree search.
	// std::cout << "Instantiating a search tree for point cloud spacing ..." << std::endl;
	Tree tree(
		boost::make_zip_iterator(boost::make_tuple(points.begin(),indices.begin())),
		boost::make_zip_iterator(boost::make_tuple(points.end(),indices.end()))
	);


#ifdef CGAL_LINKED_WITH_TBB
	tree.build<CGAL::Parallel_tag>();

	// std::cout << "Starting average spacing computation(Parallel) ..." << std::endl;
	tbb::mutex mutex;
	tbb::parallel_for (tbb::blocked_range<std::size_t> (0, num_images),
					[&](const tbb::blocked_range<std::size_t>& r) {
					for (std::size_t s = r.begin(); s != r.end(); ++ s) {
						// Neighbor search can be instantiated from
						// several threads at the same time
						const auto &query = points[s];
						K_neighbor_search search(tree, query, nb_neighbors + 1);

						const auto image_id1 = indices[s];
						std::size_t k = 0;
						Distance tr_dist;
						for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
						{
							double point_spacing = tr_dist.inverse_of_transformed_distance(it->second);

							const auto image_id2 = it->first.get<1>();
							if (image_id2 != image_id1){
								auto pair_id = sensemap::utility::ImagePairToPairId(image_id1, image_id2);
								mutex.lock();
								dist_between_images[pair_id] = point_spacing;
								mutex.unlock();
							}
						}
					}
					});
	// std::cout << "dist_between_images: " << dist_between_images.size() << std::endl;
#endif
	image_pairs.reserve(dist_between_images.size());

	uint32_t maxid = 0;
	for (const auto corres : dist_between_images) {
		uint64_t image_pair_id = corres.first;
		double dist = corres.second;
		if (dist > 0) {
			uint32_t image_id1, image_id2;
			sensemap::utility::PairIdToImagePair(
				image_pair_id, 
				&image_id1, 
				&image_id2);
			if (!valid_image_ids.empty() && 
				(valid_image_ids.find(image_id1) == valid_image_ids.end() || 
				valid_image_ids.find(image_id2) == valid_image_ids.end())){
				continue;
			}
			// image_pairs.emplace_back(image_id1, image_id2, std::sqrt(1));

			maxid = std::max(maxid, std::max(image_id1, image_id2));
			image_pairs.emplace_back(image_id1, image_id2, std::sqrt(dist));
		}
	}
	
	// Clean clusters.
	{
		std::vector<int> par(maxid + 1);
		std::vector<int> rank(maxid + 1, 1);
		std::iota(par.begin(), par.end(), 0);
		int max_rank = 0;
		int root_id = -1;
		size_t num_image_pair = image_pairs.size();
		for (size_t i = 0; i < num_image_pair; ++i) {
			uint32_t image_id1 = std::get<0>(image_pairs[i]);
			uint32_t image_id2 = std::get<1>(image_pairs[i]);
			UnionSet(image_id1, image_id2, par, rank);

			int x = Find(image_id1, par);
			if (rank[x] > max_rank) {
				max_rank = rank[x];
				root_id = x;
			}
		}
		std::cout << "max rank: " << max_rank << std::endl;
		size_t new_num_image_pair = 0;
		for (size_t i = 0; i < num_image_pair; ++i) {
			uint32_t image_id1 = std::get<0>(image_pairs[i]);
			uint32_t image_id2 = std::get<1>(image_pairs[i]);
			int x = Find(image_id1, par);
			if (x == root_id) {
				image_pairs[new_num_image_pair] = image_pairs[i];
				new_num_image_pair += 1;
			}
		}
		image_pairs.resize(new_num_image_pair);
		std::cout << sensemap::StringPrintf(
			"Pruning %d/%d image pairs", new_num_image_pair, num_image_pair) 
				  << std::endl;
	}

	return;
}

void ConvertSceneGraph2Pair(
	const std::shared_ptr<sensemap::SceneGraphContainer>& scene_graph_container,
	const std::set<uint32_t>& valid_image_ids,
	std::vector<std::tuple<uint32_t, uint32_t, double> >& image_pairs){
	
	const std::shared_ptr<sensemap::CorrespondenceGraph> &correspondence_graph = 
		scene_graph_container->CorrespondenceGraph();
	std::unordered_map<uint64_t, uint32_t> corrs_between_images =
		correspondence_graph->NumCorrespondencesBetweenImages();
	
	// std::vector<std::tuple<uint32_t, uint32_t, double> > image_pairs;
	// std::vector<float> num_inliers;
	image_pairs.reserve(corrs_between_images.size());

	uint32_t maxid = 0;
	for (const auto corres : corrs_between_images) {
		uint64_t image_pair_id = corres.first;
		uint32_t num_corr = corres.second;
		if (num_corr > 0) {
			uint32_t image_id1, image_id2;
			sensemap::utility::PairIdToImagePair(
				image_pair_id, 
				&image_id1, 
				&image_id2);
			if (!valid_image_ids.empty() && 
				(valid_image_ids.find(image_id1) == valid_image_ids.end() || 
				valid_image_ids.find(image_id2) == valid_image_ids.end())){
				continue;
			}
			// image_pairs.emplace_back(image_id1, image_id2, std::sqrt(1));

			maxid = std::max(maxid, std::max(image_id1, image_id2));
#if 0
			image_pairs.emplace_back(image_id1, image_id2, std::sqrt(num_corr));
			// num_inliers.emplace_back(num_corr);
#else
			uint32_t score = 0;
			{
				const auto& image1_scene = scene_graph_container->Image(image_id1);
				const auto& image2_scene = scene_graph_container->Image(image_id2);

				const auto& image1 = correspondence_graph->Image(image_id1);
				size_t num_corr_sum = 0;
				std::vector<Point_2> points1, points2;
				points1.reserve(num_corr);
				points2.reserve(num_corr);
				for (int pnt_idx = 0; pnt_idx < image1.corrs.size(); pnt_idx++){
					if (image1.corrs.at(pnt_idx).size() < 2) {
						continue;
					}

					for (int j = 0; j < image1.corrs.at(pnt_idx).size();j++){
						if (image1.corrs.at(pnt_idx).at(j).image_id == image_id2){
							const auto& point1 = image1_scene.Point2D(pnt_idx);
							points1.push_back(Point_2(point1.X(), point1.Y()));

							const auto& point2 = image2_scene.Point2D(image1.corrs.at(pnt_idx).at(j).point2D_idx);
							points2.push_back(Point_2(point2.X(), point2.Y()));

							num_corr_sum++;
						}
					}
				}
				if (points1.size() < 3 || points2.size() < 3 ){
					continue;
				}

				double area1 = 0;
				std::vector<std::size_t> indices1(points1.size()), out1;
				std::iota(indices1.begin(), indices1.end(),0);
				CGAL::convex_hull_2(indices1.begin(), indices1.end(), std::back_inserter(out1),
									Convex_hull_traits_2(CGAL::make_property_map(points1)));
				for (int i = 1; i < out1.size() - 1; i++){
					Eigen::Matrix3d matrix;
					matrix << points1[out1[0]].x(), points1[out1[0]].y(), 1.0f,
							points1[out1[i]].x(), points1[out1[i]].y(), 1.0f,
							points1[out1[i+1]].x(), points1[out1[i+1]].y(), 1.0f;
					area1 += std::abs(matrix.determinant()) / 2;
				}
				const auto& camera1_scene = scene_graph_container->Camera(image1_scene.CameraId());
				uint32_t ratio1 = (float)area1 / (float)(camera1_scene.Width() * camera1_scene.Height()) * 100;

				double area2 = 0;
				std::vector<std::size_t> indices2(points2.size()), out2;
				std::iota(indices2.begin(), indices2.end(),0);
				CGAL::convex_hull_2(indices2.begin(), indices2.end(), std::back_inserter(out2),
									Convex_hull_traits_2(CGAL::make_property_map(points2)));
				for (int i = 1; i < out2.size() - 1; i++){
					Eigen::Matrix3d matrix;
					matrix << points2[out2[0]].x(), points2[out2[0]].y(), 1.0f,
							points2[out2[i]].x(), points2[out2[i]].y(), 1.0f,
							points2[out2[i+1]].x(), points2[out2[i+1]].y(), 1.0f;
					area2 += std::abs(matrix.determinant()) / 2;
				}

				if (area1 < 1 || area2 < 1){
					continue;
				}
				const auto& camera2_scene = scene_graph_container->Camera(image2_scene.CameraId());
				uint32_t ratio2 = (float)area2 / (float)(camera2_scene.Width() * camera2_scene.Height()) * 1000;

				score = ratio1 * ratio2;
				// score = ratio1 * ratio2 * dist_between_images[image_pair_id];
			}
			image_pairs.emplace_back(image_id1, image_id2, std::sqrt(score));
#endif
		}
	}
	
	// Clean clusters.
	{
		std::vector<int> par(maxid + 1);
		std::vector<int> rank(maxid + 1, 1);
		std::iota(par.begin(), par.end(), 0);
		int max_rank = 0;
		int root_id = -1;
		size_t num_image_pair = image_pairs.size();
		for (size_t i = 0; i < num_image_pair; ++i) {
			uint32_t image_id1 = std::get<0>(image_pairs[i]);
			uint32_t image_id2 = std::get<1>(image_pairs[i]);
			UnionSet(image_id1, image_id2, par, rank);

			int x = Find(image_id1, par);
			if (rank[x] > max_rank) {
				max_rank = rank[x];
				root_id = x;
			}
		}
		std::cout << "max rank: " << max_rank << std::endl;
		size_t new_num_image_pair = 0;
		for (size_t i = 0; i < num_image_pair; ++i) {
			uint32_t image_id1 = std::get<0>(image_pairs[i]);
			uint32_t image_id2 = std::get<1>(image_pairs[i]);
			int x = Find(image_id1, par);
			if (x == root_id) {
				image_pairs[new_num_image_pair] = image_pairs[i];
				new_num_image_pair += 1;
			}
		}
		image_pairs.resize(new_num_image_pair);
		std::cout << sensemap::StringPrintf(
			"Pruning %d/%d image pairs", new_num_image_pair, num_image_pair) 
				  << std::endl;
	}

	return;
}

bool ConvertSceneGraph2Pose(
	const std::shared_ptr<sensemap::SceneGraphContainer>& scene_graph_container,
	const std::set<uint32_t>& valid_image_ids,
	std::unordered_map<uint32_t, Eigen::Vector3d> & image_poses){
	image_poses.clear();

	const auto& images = scene_graph_container->Images();
	const int num_image = valid_image_ids.empty() ? images.size() : valid_image_ids.size();
	image_poses.reserve(num_image);

	for (const auto& image : images){
		if (!valid_image_ids.empty() && 
			valid_image_ids.find(image.first) == valid_image_ids.end()){
			continue;
		}
		if (!image.second.HasTvecPrior()){
			continue;
		}
		image_poses[image.first] = image.second.TvecPrior();
	}

	if (image_poses.size() == 0){
		return false;
	}
	return true;
}

float AverageDist(const std::unordered_map<uint32_t, Eigen::Vector3d> &image_pose){
	if (image_pose.empty()){
		return 0.0f;
	}
	int nb_neighbors = 6;
	const size_t num = image_pose.size();
	std::vector<float> point_spacings(num, 0.f);
	double average_spacing = 0;

	std::vector<Point_3> points(num);
  	std::vector<size_t> indices(num);
	size_t i = 0;
	for (const auto image: image_pose){
		auto tvec = image.second;
		points[i] = Point_3(tvec.x(), tvec.y(), tvec.z());
		indices[i] = image.first;
		i++;
	}
	// Instantiate a KD-tree search.
	std::cout << "Instantiating a search tree for point cloud spacing ..." << std::endl;
	Tree tree(
		boost::make_zip_iterator(boost::make_tuple(points.begin(),indices.begin())),
		boost::make_zip_iterator(boost::make_tuple(points.end(),indices.end()))
	);


#ifdef CGAL_LINKED_WITH_TBB
	tree.build<CGAL::Parallel_tag>();

	std::cout << "Starting average spacing computation(Parallel) ..." << std::endl;
	tbb::parallel_for (tbb::blocked_range<std::size_t> (0, num),
					[&](const tbb::blocked_range<std::size_t>& r) {
					for (std::size_t s = r.begin(); s != r.end(); ++ s) {
						// Neighbor search can be instantiated from
						// several threads at the same time
						const auto &query = points[s];
						K_neighbor_search search(tree, query, nb_neighbors + 1);

                        auto &point_spacing = point_spacings[s];
                        point_spacing = 0.0f;
						std::size_t k = 0;
						Distance tr_dist;
						for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
						{
							point_spacing += tr_dist.inverse_of_transformed_distance(it->second);
						}
						if (k > 1) {
                        	point_spacing /= (k - 1);
                        }
					}
					});
	for (auto & point_spacing : point_spacings) {
      	average_spacing += point_spacing;
    }
#else
	const int num_eff_threads = sensemap::GetEffectiveNumThreads(-1);
	std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
	std::unique_ptr<ThreadPool> thread_pool;
	thread_pool.reset(new ThreadPool(num_eff_threads));
	{
		std::cout << "Starting average spacing computation ..." << std::endl;
		auto ComputePointSpace = [&](std::size_t i) {
			const auto &query = points[i];
			// performs k + 1 queries (if unique the query point is
			// output first). search may be aborted when k is greater
			// than number of input points
			Distance tr_dist;
			K_neighbor_search search(tree, query, nb_neighbors + 1);
			auto &point_spacing = point_spacings[i];
			point_spacing = 0.0f;
			std::size_t k = 0;
			for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
			{
				point_spacing += tr_dist.inverse_of_transformed_distance(it->second);
			}
			// output point spacing
			if (k > 1) {
				point_spacing /= (k - 1);
			}
		};

		for (std::size_t i = 0; i < num; ++i) {
			thread_pool->AddTask(ComputePointSpace, i);
		}
		thread_pool->Wait();

		for (auto & point_spacing : point_spacings) {
			average_spacing += point_spacing;
		}
	}
#endif
	
	average_spacing /= num;
	// std::cout << "Average spacing: " << average_spacing << std::endl;
	return (float)average_spacing;
};

float BetweenCommunityDist(const std::vector<uint32_t>& ref_image_ids, 
				   const std::vector<uint32_t>& src_image_ids,
				   const std::unordered_map<uint32_t, Eigen::Vector3d> &image_pose){
	int nb_neighbors = 6;
	const size_t ref_num = ref_image_ids.size();
	const size_t src_num = src_image_ids.size();
	std::vector<float> point_spacings(src_num, 0.f);
	double average_spacing = 0;

	std::vector<Point_3> ref_points(ref_num);
  	std::vector<size_t> ref_indices(ref_num);
	size_t i = 0;
	for (const auto ref_id : ref_image_ids){
		if (image_pose.find(ref_id) == image_pose.end()){
			continue;
		}
		auto tvec = image_pose.at(ref_id);
		ref_points[i] = Point_3(tvec.x(), tvec.y(), tvec.z());
		ref_indices[i] = ref_id;
		i++;
	}
	// Instantiate a KD-tree search.
	// std::cout << "Instantiating a search tree for point cloud spacing ..." << std::endl;
	Tree tree(
		boost::make_zip_iterator(boost::make_tuple(ref_points.begin(),ref_indices.begin())),
		boost::make_zip_iterator(boost::make_tuple(ref_points.end(),ref_indices.end()))
	);

	std::vector<Point_3> src_points(src_num);
  	std::vector<size_t> src_indices(src_num);
	
	size_t src_i;
	for (const auto src_id : src_image_ids){
		if (image_pose.find(src_id) == image_pose.end()){
			continue;
		}
		auto tvec = image_pose.at(src_id);
		src_points[src_i] = Point_3(tvec.x(), tvec.y(), tvec.z());
		src_indices[src_i] = src_id;
		i++;
	}
#ifdef CGAL_LINKED_WITH_TBB
	tree.build<CGAL::Parallel_tag>();

	// std::cout << "Starting average spacing computation(Parallel) ..." << std::endl;
	tbb::parallel_for (tbb::blocked_range<std::size_t> (0, src_num),
					[&](const tbb::blocked_range<std::size_t>& r) {
					for (std::size_t s = r.begin(); s != r.end(); ++ s) {
						// Neighbor search can be instantiated from
						// several threads at the same time
						const auto &query = src_points[s];
						K_neighbor_search search(tree, query, nb_neighbors + 1);

                        auto &point_spacing = point_spacings[s];
                        point_spacing = 0.0f;
						std::size_t k = 0;
						Distance tr_dist;
						for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
						{
							point_spacing += tr_dist.inverse_of_transformed_distance(it->second);
						}
						if (k > 1) {
                        	point_spacing /= (k - 1);
                        }
					}
					});
	for (auto & point_spacing : point_spacings) {
      	average_spacing += point_spacing;
    }
#else
	const int num_eff_threads = GetEffectiveNumThreads(-1);
	// std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
	std::unique_ptr<ThreadPool> thread_pool;
	thread_pool.reset(new ThreadPool(num_eff_threads));
	{
		// std::cout << "Starting average spacing computation ..." << std::endl;
		auto ComputePointSpace = [&](std::size_t i) {
			const auto &query = src_points[i];
			// performs k + 1 queries (if unique the query point is
			// output first). search may be aborted when k is greater
			// than number of input points
			Distance tr_dist;
			K_neighbor_search search(tree, query, nb_neighbors + 1);
			auto &point_spacing = point_spacings[i];
			point_spacing = 0.0f;
			std::size_t k = 0;
			for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
			{
				point_spacing += tr_dist.inverse_of_transformed_distance(it->second);
			}
			// output point spacing
			if (k > 1) {
				point_spacing /= (k - 1);
			}
		};

		for (std::size_t i = 0; i < src_num; ++i) {
			thread_pool->AddTask(ComputePointSpace, i);
		}
		thread_pool->Wait();

		for (auto & point_spacing : point_spacings) {
			average_spacing += point_spacing;
		}
	}
#endif

	average_spacing /= src_num;
	// std::cout << "Average spacing: " << average_spacing << std::endl;
	return (float)average_spacing;
}

void CommunityDetection(
	const sensemap::SceneClustering::Options& options,
	const std::shared_ptr<sensemap::SceneGraphContainer>& scene_graph_container,
	const std::set<uint32_t>& valid_image_ids,
	std::vector<std::vector<uint32_t> > &clusters,
	std::vector<std::unordered_set<uint32_t> > &overlaps) {

	std::vector<Cluster*> clusters_;
	std::vector<std::tuple<uint32_t, uint32_t, double> > image_pairs;
	ConvertSceneGraph2Pair(scene_graph_container, valid_image_ids, image_pairs);

	std::unordered_map<uint32_t, Eigen::Vector3d> image_poses;
	bool has_pose = ConvertSceneGraph2Pose(scene_graph_container, valid_image_ids, image_poses);

#if 0
	FastCommunityRecurisive(image_pairs, clusters_, options);
	std::cout << "FastCommunityRecurisive Done..., size: " << clusters_.size() << std::endl;
#else

	std::unordered_map<uint32_t, std::vector<uint32_t>> camera_communities;
    // const auto& cameras = scene_graph_container_->Cameras();
    for (const auto& image : scene_graph_container->Images()){
        const auto camre_id = image.second.CameraId();
        camera_communities[camre_id].push_back(image.first);
    }


	std::vector<std::pair<uint32_t, std::vector<uint32_t>>> camera_clusters;
	std::vector<uint8_t> cross_merge_flags;
    for (const auto& camera_community : camera_communities){
        if (camera_community.second.size() > options.max_modularity_count){
            std::set<uint32_t> valid_camera_image_ids;
            valid_camera_image_ids.insert(camera_community.second.begin(), camera_community.second.end());

			std::vector<std::tuple<uint32_t, uint32_t, double> > community_image_pairs;
			ConvertSceneGraph2Pair(scene_graph_container, valid_camera_image_ids, community_image_pairs);

			std::unordered_map<uint32_t, Eigen::Vector3d> community_image_poses;
            ConvertSceneGraph2Pose(scene_graph_container, valid_image_ids, community_image_poses); 

			std::vector<Cluster*> clusters;
			FastCommunityRecurisive(community_image_pairs, clusters, options);
			MergeClusters(community_image_pairs, community_image_poses, clusters, options);
			MergePieces(community_image_pairs, clusters, options);

			std::sort(clusters.begin(), clusters.end(), 
            	[](const Cluster* a, const Cluster* b){return a->image_ids.size() > b->image_ids.size(); });

			for (int i = 0; i < clusters.size(); i++){
				camera_clusters.push_back(std::make_pair(
					camera_community.first, clusters.at(i)->image_ids));
				if (i == 0){
					cross_merge_flags.push_back((uint8_t)2);
				} else{
					cross_merge_flags.push_back((uint8_t)1);
				}
				std::cout << "add Camera-" << camera_community.first << " cluster-" << i 
					<< " => " << clusters.at(i)->image_ids.size() << ", "
					<< (int)clusters.at(i)->cross_merge << std::endl;
			}
        } else {
			camera_clusters.push_back(camera_community);
			cross_merge_flags.push_back((uint8_t)0);
			std::cout << "add Camera-" << camera_community.first << " => "<< camera_community.second.size() << std::endl;
		}
	}

	clusters_.clear();
	clusters_.shrink_to_fit();
	int cluster_idx = 0;
	// for (const auto& camera_clsuter : camera_clusters){
	for (uint32_t i = 0; i < camera_clusters.size(); i++){
		const auto& camera_clsuter = camera_clusters.at(i);
		clusters_.insert(clusters_.end(), new Cluster());
		clusters_.at(cluster_idx)->cluster_id = cluster_idx;
		clusters_.at(cluster_idx)->image_ids = camera_clsuter.second;
		clusters_.at(cluster_idx)->cross_merge = cross_merge_flags.at(i);
		cluster_idx++;
	}

	std::cout << "clusters_: " << clusters_.size() << ", camera_clusters:" << camera_clusters.size() << std::endl;
#endif

#if 1

	MergeClusters(image_pairs, image_poses, clusters_, options);

	MergePieces(image_pairs, clusters_, options);

	// AddOverlapImages(image_pairs, num_inliers, neighbors, 
	// 	clusters_, overlaps, options);

	clusters.resize(clusters_.size());
	for (int i = 0; i < clusters.size(); ++i) {
		clusters[i] = clusters_[i]->image_ids;
		delete clusters_[i];
		clusters_[i] = nullptr;
	}
	std::vector<Cluster*>().swap(clusters_);
#else
	auto root = new CommunityTree<uint32_t>();
	FastBuildCommunityTree(image_pairs, root, options);

#if 1 //output community struct
	root->writeDot("./graph.dot");

    std::ofstream file("./graph.dot", std::ios::app);
	auto leaves = root->GetLeaves();
	for(auto leaf : leaves){
        file << std::endl;
        file << "subgraph cluster_" << leaf->Id() << " { " << std::endl;
        file << "   label = \"community "<< leaf->Id()  << "\";" <<std::endl;
        for(auto element : leaf->elements()){
            file << element << " ";
        }
        file << std::endl;
        file << "}" <<std::endl;
	}
	file.close();
#endif

    std::list<CommunityTree<uint32_t> *> cluster_trees;
    cluster_trees = MergeCommunityTree(root, options);
    for(auto cluster_tree : cluster_trees){
        clusters.push_back(cluster_tree->elements());
    }
#endif
}

void CommunityAddOverlap(
	const sensemap::SceneClustering::Options& options,
	const std::shared_ptr<sensemap::CorrespondenceGraph> &correspondence_graph,
	std::vector<std::vector<uint32_t> > &clusters,
	std::vector<std::unordered_set<uint32_t> > &overlaps){
		
	std::vector<Cluster*> clusters_;

	const std::unordered_map<uint32_t, std::unordered_set<uint32_t> >&
		neighbors = correspondence_graph->ImageNeighbors();

	std::unordered_map<uint64_t, uint32_t> corrs_between_images =
		correspondence_graph->NumCorrespondencesBetweenImages();

	std::vector<std::tuple<uint32_t, uint32_t, double> > image_pairs;
	std::vector<float> num_inliers;
	for (const auto corres : corrs_between_images) {
		uint64_t image_pair_id = corres.first;
		uint32_t num_corr = corres.second;
		if (num_corr > 0) {
			uint32_t image_id1, image_id2;
			sensemap::utility::PairIdToImagePair(
				image_pair_id, 
				&image_id1, 
				&image_id2);
			image_pairs.emplace_back(image_id1, image_id2, std::sqrt(num_corr));
			num_inliers.emplace_back(num_corr);
		}
	}

	// Set value for clusters_
	for (int i = 0; i < clusters.size(); ++i) {
		Cluster *cluster = new Cluster();
		cluster->cluster_id = i;
		cluster->image_ids = clusters[i];
		clusters_.emplace_back(cluster);
	}

	AddOverlapImages(image_pairs, num_inliers, neighbors, 
					 clusters_, overlaps, options);

	for (int i = 0; i < clusters.size(); ++i) {
		clusters[i] = clusters_[i]->image_ids;
		delete clusters_[i];
		clusters_[i] = nullptr;
	}
	std::vector<Cluster*>().swap(clusters_);

}


//----------------------------------------------------------------------------//
// FUNCTION DEFINITIONS 
//----------------------------------------------------------------------------//
namespace {

int Find(int x, std::vector<int>& par) {
	if (x != par[x] && par[x] != -1) {
		par[x] = Find(par[x], par);
	}
	return par[x];
}

void UnionSet(int x, int y, std::vector<int>& par, std::vector<int>& rank) {
	x = Find(x, par);
	y = Find(y, par);
	if (x == y) {
		return;
	}
	if (rank[x] < rank[y]) {
		par[x] = y;
		rank[y] += rank[x];
	} else {
		par[y] = x;
		rank[x] += rank[y];
	}
}

void BuildLabels(int t, std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options) {
	if (t <= 0) return ;

	int level;
	double Qmax;
	std::set<int> clusters_id;
	std::vector<int> par(gparm.maxid, -1);
	// int *par = new int[gparm.maxid];
	// memset(par, -1, sizeof(int) * gparm.maxid);

	for (level = t - 1, Qmax = Q[level]; level > 0; --level) {
		std::cout << "level = " << level << ", " << Q[level] << " / " << Qmax << std::endl;
		if (Q[level] < Q[level - 1]) {
			Qmax = Q[level - 1];
			clusters_id.insert(joins[level + 1].x);
			clusters_id.insert(joins[level + 1].y);
		} else {
			break;
		}
	}

	if (clusters_id.size() == 0 || Qmax < options.min_modularity_thres) {
		clusters_id.clear();
		clusters_id.insert(joins[t].y);
		level = t - 1;
	}

	std::cout << "cluster id : " << std::endl;
	for (const auto & cluster_id : clusters_id) {
		par[cluster_id] = cluster_id;
		std::cout << cluster_id << " ";
	}
	std::cout << std::endl;

	for (; level >= 0; --level) {
		// int x = par[joins[level + 1].x];
		int y = par[joins[level + 1].y];
		y = Find(y, par);
		par[joins[level + 1].x] = y;
	}

	std::unordered_map<int, std::vector<uint32_t> > cluster_id_map;
	for (int i = 0; i < gparm.maxid; ++i) {
		int id = Find(i, par);
		if (id != -1) {
			cluster_id_map[id].push_back(i);
		}
	}
	for (const auto & mp : cluster_id_map) {
		Cluster *cluster = new Cluster();
		cluster->cluster_id = mp.first;
		cluster->image_ids = mp.second;
		clusters.emplace_back(cluster);
		std::cout << "cluster : " << mp.first << ", " 
				  << mp.second.size() << std::endl;
	}
}

void MergeClusters(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	const std::unordered_map<uint32_t, Eigen::Vector3d> &image_pose,
	std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options) {

// DEBUG
#if 1
	std::unordered_set<uint32_t> image_id_set1;
	for (const auto & pair_id : image_pair_ids) {
		image_id_set1.insert(std::get<0>(pair_id));
		image_id_set1.insert(std::get<1>(pair_id));
	}
	std::unordered_set<uint32_t> image_id_set2;
	for (const auto & cluster : clusters) {
		image_id_set2.insert(cluster->image_ids.begin(), 
			cluster->image_ids.end());
	}
	std::cout << "number of images from image_pair: " << image_id_set1.size() 
			  << std::endl;
	std::cout << "number of images from " << clusters.size() << " clusters  : " << image_id_set2.size() 
			  << std::endl;
#endif

	clusters.insert(clusters.begin(), new Cluster());

	int i, j;
	const size_t num_cluster = clusters.size();

	std::unordered_map<uint32_t, int> image_cluster_id_map;
	for (i = 0; i < num_cluster; ++i) {
		for (const auto & image_id : clusters[i]->image_ids) {
			image_cluster_id_map[image_id] = i;
		}
	}

	std::vector<std::vector<int> > num_inter_clusters(
		num_cluster, std::vector<int>(num_cluster, 0));

	std::vector<std::vector<int> > num_inter_clusters2(num_cluster, 
		std::vector<int>(num_cluster, 0));
	std::vector<std::vector<std::unordered_set<uint32_t>> > cluster_neighbor_images(
		num_cluster, std::vector<std::unordered_set<uint32_t>>(num_cluster));

	std::vector<std::vector<int> > num_inter_clusters3(num_cluster, 
		std::vector<int>(num_cluster, 0));
	std::vector<std::vector<float> > cluster_neighbor_dist(
		num_cluster, std::vector<float>(num_cluster));

	float average_dist = AverageDist(image_pose);
	// std::vector<std::vector<std::unordered_set<float>> > cluster_neighbor_dist(
	// 	num_cluster, std::vector<std::unordered_set<float>>(num_cluster));

	for (auto image_pair_id : image_pair_ids) {
		if (image_cluster_id_map.count(std::get<0>(image_pair_id)) &&
			image_cluster_id_map.count(std::get<1>(image_pair_id))) {
			uint32_t image_id1 = std::get<0>(image_pair_id);
			uint32_t image_id2 = std::get<1>(image_pair_id);
			int cluster_id1 = image_cluster_id_map[image_id1];
			int cluster_id2 = image_cluster_id_map[image_id2];
			if (cluster_id1 != cluster_id2 && 
				(clusters.at(cluster_id1)->cross_merge + clusters.at(cluster_id2)->cross_merge) < 3) {
				num_inter_clusters[cluster_id1][cluster_id2]++;
				num_inter_clusters[cluster_id2][cluster_id1]++;

				cluster_neighbor_images[cluster_id1][cluster_id2].insert(image_id2);
				cluster_neighbor_images[cluster_id2][cluster_id1].insert(image_id1);
			}
		}
	}

	for (size_t idx1 = 0; idx1 < num_inter_clusters.size(); idx1++){
		auto& clusters_neighbor = num_inter_clusters.at(idx1);
		for (size_t idx2 = idx1; idx2 < clusters_neighbor.size(); idx2++){
			if (num_inter_clusters[idx1][idx2] == 0) {
				continue;
			}
			auto min_image_size = std::min(clusters[idx1]->image_ids.size(), clusters[idx2]->image_ids.size());
			// auto num_neig = num_inter_clusters[idx1][idx2];
			// num_inter_clusters[idx1][idx2] = num_neig * 10 / min_image_size;

			float score_cluster_idx1 = (float)cluster_neighbor_images[idx2][idx1].size() 
										/ (float)clusters[idx1]->image_ids.size();
			float score_cluster_idx2 = (float)cluster_neighbor_images[idx1][idx2].size() 
										/ (float)clusters[idx2]->image_ids.size();
			float score_neighbor_images = std::max(score_cluster_idx1, score_cluster_idx2) * 10000.0f + 0.5f;
			// num_inter_clusters2[idx1][idx2] = score_neighbor_images;
			// num_inter_clusters2[idx2][idx1] = score_neighbor_images;

			float score_dist = 0;
			if (!image_pose.empty()){
				float dist_1t2 = BetweenCommunityDist(clusters[idx1]->image_ids, 
											clusters[idx2]->image_ids, image_pose);
				float dist_2t1 = BetweenCommunityDist(clusters[idx2]->image_ids, 
											clusters[idx1]->image_ids, image_pose);
				score_dist = average_dist / std::min(dist_1t2, dist_2t1) * 10000.0f + 0.5f;
			}

			num_inter_clusters2[idx1][idx2] = score_neighbor_images + score_dist;
			num_inter_clusters2[idx2][idx1] = score_neighbor_images + score_dist;
			// std::cout << "cluster " << idx1 << " - " << idx2 << ": " << dist_1t2 << ", "
			// 	<< dist_2t1 << ", " << average_dist << ", " << score_dist << ", " 
			// 	<< score_neighbor_images << std::endl;
		}
	}
	// exit(-1);

	std::vector<bool> valid_cluster(num_cluster, true);

	nodenub* btree = new nodenub[num_cluster];
	maxheap* max_heap = new maxheap();

	btree[0].v = NULL;
	for (i = 1; i < num_cluster; ++i) {
		valid_cluster[i] = 
			clusters[i]->image_ids.size() >= options.min_modularity_count;

#ifdef VERBOSE
		std::cout << "cluster#" << i << " size = " 
				  << clusters[i]->image_ids.size() << "/" 
				  << options.min_modularity_count << ", valid = " 
				  << valid_cluster[i] << std::endl;
#endif

		tuple max_connection;
		max_connection.i = i;
		max_connection.m = 0.0;

		std::vector<uint32_t> neighbors;
		for (j = 0; j < num_cluster; ++j) {
			if ((clusters[i]->image_ids.size() < options.min_modularity_count ||
				clusters[j]->image_ids.size() < options.min_modularity_count) &&
				num_inter_clusters2[i][j] > 0) {
				neighbors.emplace_back(j);
				std::cout << "neighbors emplace_back: " << j << ", " << clusters[j]->image_ids.size() << std::endl;
			}
		}

		btree[i].v = new vektor(neighbors.size());
		for (const auto & neighbor : neighbors) {
#ifdef VERBOSE
			std::cout << "	" << neighbor << ", "
			<< num_inter_clusters2[i][neighbor] << " | ";
#endif
			btree[i].v->insertItem(neighbor, num_inter_clusters2[i][neighbor]);
			if (num_inter_clusters2[i][neighbor] > max_connection.m) {
				max_connection.j = neighbor;
				max_connection.m = num_inter_clusters2[i][neighbor];
			}
		}
		if (neighbors.size() > 0) {
#ifdef VERBOSE
			std::cout << std::endl << "	init max_connection " << max_connection.i << ", " 
				  << max_connection.j << ", " << max_connection.m << std::endl;
#endif
			btree[i].heap_ptr = max_heap->insertItem(max_connection);
		}
	}

	while(max_heap->heapSize() >= 1) {

#ifdef VERBOSE
		max_heap->printHeap();
#endif

		tuple max_connection = max_heap->popMaximum();
#ifdef VERBOSE
		std::cout << "heap max_connection " << max_connection.i << ", " 
					<< max_connection.j << ", " << max_connection.m << std::endl;
#endif
		int cluster_id1 = max_connection.i;
		int cluster_id2 = max_connection.j;
		bool condition0 = (cluster_id1 == 0 || cluster_id2 == 0);
		bool condition1 = (!btree[cluster_id1].v || !btree[cluster_id2].v);
		bool condition2 = (valid_cluster[cluster_id1] && valid_cluster[cluster_id2]);  
		bool condition3 = (clusters[cluster_id1]->image_ids.size() + clusters[cluster_id2]->image_ids.size()) > options.max_modularity_count;
		if (condition0 || condition1 || condition2 || condition3) {
			continue;
		} else if (clusters[cluster_id1]->image_ids.size() < 
				   clusters[cluster_id2]->image_ids.size()) {
			std::swap(cluster_id1, cluster_id2);
		}

		if (clusters[cluster_id1]->image_ids.size() >= options.max_modularity_count) {
			dpair* list = btree[cluster_id1].v->returnTreeAsList();
			dpair* current = list;
			while(current != NULL) {
				if (cluster_id1 != current->x) {
					btree[current->x].v->deleteItem(cluster_id1);
					tuple new_max = btree[current->x].v->returnMaxStored();
					if (current->x == max_connection.i) {
						new_max.i = current->x;
						btree[current->x].heap_ptr = max_heap->insertItem(new_max);
					} else {
						max_heap->updateItem(btree[current->x].heap_ptr, new_max);
					}
				}
				current = current->next;
			}

			// free memory
			delete btree[cluster_id1].v;
			btree[cluster_id1].v = NULL;
		} else {
			// merge cluster_id2 to cluster_id1.
			dpair* list = btree[cluster_id2].v->returnTreeAsList();
			dpair* current = list;
			while(current != NULL) {
				if (cluster_id1 != current->x) {
					if (!(valid_cluster[cluster_id1] && valid_cluster[current->x])) {
						// btree[cluster_id1].v->insertItem(current->x, current->y);
						// btree[current->x].v->insertItem(cluster_id1, current->y);
						cluster_neighbor_images[cluster_id1][current->x].insert(
							cluster_neighbor_images[current->x][cluster_id2].begin(),
							cluster_neighbor_images[current->x][cluster_id2].end());
						cluster_neighbor_images[current->x][cluster_id1].insert(
							cluster_neighbor_images[current->x][cluster_id2].begin(),
							cluster_neighbor_images[current->x][cluster_id2].end());
						
						float score_cluster_idx1 = 
							(float)cluster_neighbor_images[current->x][cluster_id1].size() 
							/ (float)(clusters[cluster_id1]->image_ids.size() 
							+ clusters[cluster_id2]->image_ids.size());
						float score_cluster_idxCX = 
							(float)cluster_neighbor_images[current->x][cluster_id1].size() 
							/ (float)clusters[current->x]->image_ids.size();

						float score_neighbor_images = std::max(score_cluster_idx1, score_cluster_idxCX) * 10000.0f + 0.5f;

						float score_dist = 0;
						if (!image_pose.empty()){
							std::vector<uint32_t> images_vec;
							images_vec.insert(images_vec.end(), clusters[cluster_id1]->image_ids.begin(), clusters[cluster_id1]->image_ids.end());
							images_vec.insert(images_vec.end(), clusters[cluster_id2]->image_ids.begin(), clusters[cluster_id2]->image_ids.end());
							float dist_1t2 = BetweenCommunityDist(images_vec, clusters[current->x]->image_ids, image_pose);
							float dist_2t1 = BetweenCommunityDist(clusters[current->x]->image_ids, images_vec, image_pose);
							score_dist = average_dist / std::min(dist_1t2, dist_2t1) * 10000.0f + 0.5f;
						}
						

						double score = score_neighbor_images + score_dist;
						num_inter_clusters2[cluster_id1][current->x] = score;
						num_inter_clusters2[current->x][cluster_id1] = score;
						
						btree[cluster_id1].v->insertItem2(current->x, score);
						btree[current->x].v->insertItem2(cluster_id1, score);
					}
					
					btree[current->x].v->deleteItem(cluster_id2);
					tuple new_max = btree[current->x].v->returnMaxStored();
					if (current->x == max_connection.i) {
						new_max.i = current->x;
						btree[current->x].heap_ptr = max_heap->insertItem(new_max);
					} else {
						max_heap->updateItem(btree[current->x].heap_ptr, new_max);
					}
				}
				current = current->next;
			}
			
			btree[cluster_id1].v->deleteItem(cluster_id2);
			if (btree[cluster_id1].heap_ptr->m < 0){
				break;
			}
			tuple new_max = btree[cluster_id1].v->returnMaxStored();
			if (btree[cluster_id1].v->returnNodecount() > 0) {
				new_max.i = cluster_id1;
				btree[cluster_id1].heap_ptr = max_heap->insertItem(new_max);
			} else if (cluster_id1 != max_connection.i) {
				max_heap->updateItem(btree[cluster_id1].heap_ptr, new_max);
			}
	#ifdef VERBOSE
			std::cout << "cluster1.size = " 
					<< clusters[cluster_id1]->image_ids.size() << ", "
					<< "cluster2.size = " 
					<< clusters[cluster_id2]->image_ids.size() << std::endl;
			std::cout << "Merged cluster#" << cluster_id2 
					<< " to cluster#" << cluster_id1 << std::endl;
			std::cout << "max_heap.size = " << max_heap->heapSize() 
					  << std::endl;
	#endif
			clusters[cluster_id1]->image_ids.insert(
				clusters[cluster_id1]->image_ids.end(), 
				clusters[cluster_id2]->image_ids.begin(), 
				clusters[cluster_id2]->image_ids.end());
			clusters[cluster_id1]->cross_merge = 
				std::max(clusters[cluster_id1]->cross_merge, clusters[cluster_id2]->cross_merge);
			clusters[cluster_id2]->image_ids.clear();

			valid_cluster[cluster_id1] = 
				clusters[cluster_id1]->image_ids.size() >= options.min_modularity_count;

			// free memory
			delete btree[cluster_id2].v;
			btree[cluster_id2].v = NULL;
		}
		
#ifdef VERBOSE
		// print tree
		for (int i = 1; i < num_cluster; ++i) {
			if (!btree[i].v) continue;
			std::cout << "	cluster#" << i << std::endl;
			dpair* list1 = btree[i].v->returnTreeAsList();
			dpair* current1 = list1;
			std::cout << "	";
			while(current1 != NULL) {
				std::cout << " " << current1->x << ", " << current1->y;
				current1 = current1->next;
			}
			std::cout << std::endl;
		}
#endif
	}

	for (i = 0, j = 0; i < num_cluster; ++i) {
		if (!btree[i].v && clusters[i]->image_ids.empty()) {
			delete clusters[i];
			continue;
		}
		clusters[j] = clusters[i];
		clusters[j]->cluster_id = j;
		j = j + 1;

		if (btree[i].v) {
			delete btree[i].v;
			btree[i].v = NULL;
		}
	}
	clusters.resize(j);

	delete [] btree;
	delete max_heap;
	
	std::cout << "Merged " << num_cluster - clusters.size() 
			<< " / " << num_cluster << " Clusters" << std::endl;
}

void MergePieces(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options) {
	
	size_t i, j;
	const size_t num_cluster = clusters.size();

	std::sort(clusters.begin(), clusters.end(), 
		[&](const Cluster* cluster1, const Cluster* cluster2) {
		return cluster1->image_ids.size() < cluster2->image_ids.size();
	});

	std::unordered_map<uint32_t, int> image_cluster_id_map;
	for (i = 0; i < num_cluster; ++i) {
		for (const auto & image_id : clusters[i]->image_ids) {
			image_cluster_id_map[image_id] = i;
		}
	}

	std::vector<std::vector<int> > num_inter_clusters(
		num_cluster, std::vector<int>(num_cluster, 0));
	
	std::vector<std::vector<std::unordered_set<uint32_t>> > cluster_neighbor_images(
		num_cluster, std::vector<std::unordered_set<uint32_t>>(num_cluster));

	for (auto image_pair_id : image_pair_ids) {
		if (image_cluster_id_map.count(std::get<0>(image_pair_id)) &&
			image_cluster_id_map.count(std::get<1>(image_pair_id))) {
			uint32_t image_id1 = std::get<0>(image_pair_id);
			uint32_t image_id2 = std::get<1>(image_pair_id);
			int cluster_id1 = image_cluster_id_map[image_id1];
			int cluster_id2 = image_cluster_id_map[image_id2];
			if (cluster_id1 != cluster_id2 && 
				(clusters.at(cluster_id1)->cross_merge + clusters.at(cluster_id2)->cross_merge) < 3) {
				num_inter_clusters[cluster_id1][cluster_id2]++;
				num_inter_clusters[cluster_id2][cluster_id1]++;

				cluster_neighbor_images[cluster_id1][cluster_id2].insert(image_id2);
				cluster_neighbor_images[cluster_id2][cluster_id1].insert(image_id1);
			}
		}
	}

	for (size_t idx1 = 0; idx1 < num_inter_clusters.size(); idx1++){
		auto& clusters_neighbor = num_inter_clusters.at(idx1);
		for (size_t idx2 = 0; idx2 < clusters_neighbor.size(); idx2++){
			if (num_inter_clusters[idx1][idx2] <= 0) {
				continue;
			}
			// auto min_image_size = std::min(clusters[idx1]->image_ids.size(), clusters[idx2]->image_ids.size());
			// auto num_neig = num_inter_clusters[idx1][idx2];
			// num_inter_clusters[idx1][idx2] = num_neig * 10 / min_image_size;

			float score_cluster_idx1 = (float)cluster_neighbor_images[idx2][idx1].size() 
										/ (float)clusters[idx1]->image_ids.size();
			float score_cluster_idx2 = (float)cluster_neighbor_images[idx1][idx2].size() 
										/ (float)clusters[idx2]->image_ids.size();
			float min_neighbor_images = std::max(score_cluster_idx1, score_cluster_idx2);
			num_inter_clusters[idx1][idx2] = min_neighbor_images * 10000.0f + 0.5f;
		}
	}

	for (i = 0; i < num_cluster; ++i) {
		if (clusters[i]->image_ids.size() >= options.min_modularity_count) {
			continue;
		}
		
		std::cout << "cluster " << i << ", " << clusters[i]->image_ids.size() << std::endl;

		int max_connection = 0;
		int merged_idx = -1;
		for (j = i + 1; j < num_cluster; ++j) {
			if (max_connection < num_inter_clusters[i][j] && 
				(clusters[i]->image_ids.size() + clusters[j]->image_ids.size()) < options.max_modularity_count) {
				max_connection = num_inter_clusters[i][j];
				merged_idx = j;
			}
		}
		if (merged_idx < 0){
			std::cout << "no find max_connection, continue ..." << std::endl;
			continue;
		}
		std::cout << "\t merge idx : " << merged_idx << ", " << clusters[merged_idx]->image_ids.size() << std::endl;
		
		if (max_connection > 0) {
			// Update num_inter_clusters.
			for (j = 0; j < num_cluster; ++j) {
				if (merged_idx != j) {
					if (cluster_neighbor_images[merged_idx][i].empty() 
						&& cluster_neighbor_images[i][merged_idx].empty()){

						std::cout << "/t before num_inter_clusters: " << merged_idx << " - " << j 
							<< ", " << num_inter_clusters[merged_idx][j] << std::endl;

						cluster_neighbor_images[merged_idx][j].insert(
							cluster_neighbor_images[merged_idx][i].begin(), 
							cluster_neighbor_images[merged_idx][i].end());
						
						cluster_neighbor_images[j][merged_idx].insert(
							cluster_neighbor_images[i][merged_idx].begin(),
							cluster_neighbor_images[i][merged_idx].end());
							
						float score_cluster_idx1 = (float)cluster_neighbor_images[merged_idx][j].size() 
													/ (float)clusters[j]->image_ids.size();
						float score_cluster_idx2 = (float)cluster_neighbor_images[j][merged_idx].size() 
							/ (float)(clusters[merged_idx]->image_ids.size() + clusters[i]->image_ids.size());
						
						float min_neighbor_images = std::max(score_cluster_idx1, score_cluster_idx2);
						num_inter_clusters[merged_idx][j] = min_neighbor_images * 10000.0f + 0.5f;
						num_inter_clusters[j][merged_idx] = min_neighbor_images * 10000.0f + 0.5f;

						std::cout << "/t after num_inter_clusters: " << merged_idx << " - " << j 
							<< ", " << num_inter_clusters[merged_idx][j] << std::endl;
					}
				}
				num_inter_clusters[i][j] = num_inter_clusters[j][i] = 0;
			}
			
			clusters[merged_idx]->image_ids.insert(
				clusters[merged_idx]->image_ids.end(), 
				clusters[i]->image_ids.begin(), 
				clusters[i]->image_ids.end());
			clusters[merged_idx]->cross_merge = 
				std::max(clusters[i]->cross_merge, clusters[merged_idx]->cross_merge);
			clusters[i]->image_ids.clear();
			std::cout << "\t merge " << i << " => " << merged_idx << std::endl;

			delete clusters[i];
			clusters[i] = NULL;
		}
	}

	for (i = 0, j = 0; i < num_cluster; ++i) {
		if (clusters[i]) {
			clusters[j] = clusters[i];
			clusters[j]->cluster_id = j;
			j = j + 1;
		}
	}
	clusters.resize(j);
}

void GetTopKTwoCluster(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	const std::vector<float> &weights,
	const Cluster* cluster1,
	const Cluster* cluster2,
	const std::unordered_map<uint32_t, int>& image_to_cluster_id_map,
	const int image_overlap,
	std::vector<IMAGE_PAIR>& edges) {

	edges.clear();
	if (image_overlap == 0) {
		return ;
	}
	
	struct SortedImagePair {
		IMAGE_PAIR image_pair;
		float weight;
		bool const operator < (const SortedImagePair &other) const {
			return weight < other.weight;
		}
	};
	std::set<SortedImagePair> sorted_image_pairs;

	for (int i = 0; i < image_pair_ids.size(); ++i) {
		const auto& image_pair = image_pair_ids[i];
		std::unordered_map<uint32_t, int>::const_iterator it1 =
			image_to_cluster_id_map.find(std::get<0>(image_pair));
		std::unordered_map<uint32_t, int>::const_iterator it2 =
			image_to_cluster_id_map.find(std::get<1>(image_pair));
		if (it1 == image_to_cluster_id_map.end() ||
			it2 == image_to_cluster_id_map.end() ||
			it1->second == it2->second) {
			continue;
		}
		if ((it1->second != cluster1->cluster_id &&
			 it1->second != cluster2->cluster_id) ||
			(it2->second != cluster1->cluster_id &&
			 it2->second != cluster2->cluster_id)) {
			continue;
		}
		SortedImagePair sorted_image_pair;
		sorted_image_pair.image_pair.first = std::get<0>(image_pair);
		sorted_image_pair.image_pair.second = std::get<1>(image_pair);
		sorted_image_pair.weight = weights[i];
		// cluster_id1 always be in front.
		if (it1->second == cluster2->cluster_id) {
			std::swap(sorted_image_pair.image_pair.first, 
				sorted_image_pair.image_pair.second);
		}
		sorted_image_pairs.insert(sorted_image_pair);
	}

	int start = sorted_image_pairs.size() - 1;
	int end = std::max((int)sorted_image_pairs.size() - image_overlap, 0);
	std::set<SortedImagePair>::iterator it = sorted_image_pairs.begin();
	for (; start >= end; --start) {
		edges.emplace_back(it->image_pair);
		std::advance(it, 1);
	}
}

void GetSeqOverlapTwoClusters(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	const std::vector<float> &weights,
	const Cluster* cluster1,
	const Cluster* cluster2,
	const std::unordered_map<uint32_t, int>& image_to_cluster_id_map,
	const int image_dist_seq_overlap,
	std::vector<IMAGE_PAIR>& edges) {

	edges.clear();

	for (int i = 0; i < image_pair_ids.size(); ++i) {
		const auto& image_pair = image_pair_ids[i];
		uint32_t image_id0 = std::get<0>(image_pair);
		uint32_t image_id1 = std::get<1>(image_pair);
		std::unordered_map<uint32_t, int>::const_iterator it1 =
			image_to_cluster_id_map.find(image_id0);
		std::unordered_map<uint32_t, int>::const_iterator it2 =
			image_to_cluster_id_map.find(image_id1);
		if (it1 == image_to_cluster_id_map.end() ||
			it2 == image_to_cluster_id_map.end() ||
			it1->second == it2->second) {
			continue;
		}
		if ((it1->second != cluster1->cluster_id &&
			 it1->second != cluster2->cluster_id) ||
			(it2->second != cluster1->cluster_id &&
			 it2->second != cluster2->cluster_id)) {
			continue;
		}

		int image_dist = (image_id0 > image_id1) ? 
			image_id0 - image_id1 : image_id1 - image_id0;
		if (image_dist <= image_dist_seq_overlap) {
			// cluster_id1 always be in front.
			if (it1->second == cluster2->cluster_id) {
				std::swap(image_id0, image_id1);
			}
			edges.emplace_back(image_id0, image_id1);
		}
	}
}

void FindTransitivity(
	const int transitivity,
	const int cluster_id,
	const uint32_t image_id,
	const std::unordered_map<uint32_t, int>& image_to_cluster_id_map,
	const std::unordered_map<uint32_t, std::unordered_set<uint32_t> >& neighbors,
	std::vector<uint32_t>& transfer_images,
	const int image_dist_seq_overlap = INT_MAX) {
	transfer_images.clear();
	transfer_images.emplace_back(image_id);

	std::unordered_set<uint32_t> visited;
	visited.emplace(image_id);

	int start = 0;
	int end = transfer_images.size();
	for (int t = 0; t < transitivity; ++t) {
		for (int i = start; i < end; ++i) {
			const std::unordered_set<uint32_t>& neighbor_ids = 
				neighbors.at(transfer_images[i]);
			for (const auto & neighbor_id : neighbor_ids) {
				if (image_to_cluster_id_map.at(neighbor_id) == cluster_id &&
						visited.count(neighbor_id) == 0) {
					int image_dist = image_id > neighbor_id ?
						image_id - neighbor_id : neighbor_id - image_id;
					if (image_dist > image_dist_seq_overlap) {
						continue;
					}
					transfer_images.emplace_back(neighbor_id);
					visited.emplace(neighbor_id);
				}
			}
		}
		start = end;
		end = transfer_images.size();
	}
}

void AddOverlapImages(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	const std::vector<float> &weights,
	const std::unordered_map<uint32_t, std::unordered_set<uint32_t> >& neighbors,
	std::vector<Cluster*> &clusters,
	std::vector<std::unordered_set<uint32_t> > &overlaps,
	const sensemap::SceneClustering::Options& options) {
	
	const int num_cluster = clusters.size();
	overlaps.resize(num_cluster);

	if (num_cluster < 2) {
		return;
	} 

	const int image_overlap = options.community_image_overlap;
	const int image_dist_seq_overlap = options.image_dist_seq_overlap;
	const int transitivity = options.community_transitivity;

	std::vector<std::vector<GRAPH_EDGE> > graph(num_cluster);

	std::unordered_map<uint32_t, int> image_to_cluster_id_map;
	for (int cluster_id = 0; cluster_id < num_cluster; ++cluster_id) {
		std::for_each(clusters[cluster_id]->image_ids.begin(),
					  clusters[cluster_id]->image_ids.end(),
					  [&](const uint32_t& image_id) {
						image_to_cluster_id_map[image_id] = cluster_id;
					  });
		graph[cluster_id].resize(num_cluster);
		for (auto & edge : graph[cluster_id]) {
			edge.first = 0;
			edge.second = 0;
		}
		clusters[cluster_id]->cluster_id = cluster_id; //rearrange cluster id
	}

	for (const auto & image_pair : image_pair_ids) {
		std::unordered_map<uint32_t, int>::iterator it1 =
			image_to_cluster_id_map.find(std::get<0>(image_pair));
		std::unordered_map<uint32_t, int>::iterator it2 =
			image_to_cluster_id_map.find(std::get<1>(image_pair));
		if (it1 == image_to_cluster_id_map.end() ||
			it2 == image_to_cluster_id_map.end() ||
			it1->second == it2->second) {
			continue;
		}
		int cluster_id1 = it1->second;
		int cluster_id2 = it2->second;
		graph[cluster_id1][cluster_id2].first = cluster_id2;
		graph[cluster_id1][cluster_id2].second++;
		graph[cluster_id2][cluster_id1].first = cluster_id1;
		graph[cluster_id2][cluster_id1].second++;
	}

	// sort by links
	int num_link = 0;
	GRAPH_EDGE edge = std::make_pair(-1, -1);
	for (int i = 0; i < num_cluster; ++i) {
		std::sort(graph[i].begin(), graph[i].end(), 
			[](const GRAPH_EDGE& l1, const GRAPH_EDGE& l2) {
				return l1.second > l2.second;
			});
		if (num_link < graph[i][0].second) {
			num_link = graph[i][0].second;
			edge.first = i;
			edge.second = graph[i][0].first;
		}
	}
	
	std::vector<bool> visited(num_cluster, false);

	std::queue<int> Q;
	Q.push(0);
	while(!Q.empty()) {
		int cluster_id1 = Q.front();
		Q.pop();

		visited[cluster_id1] = true;

		for (int i = 0; i < graph[cluster_id1].size(); ++i) {
			int cluster_id2 = graph[cluster_id1][i].first;
			if (graph[cluster_id1][i].second > 0 && !visited[cluster_id2]) {

				Q.push(cluster_id2);

				// detect overlap according to scene graph.
				std::vector<std::pair<uint32_t, uint32_t> > edges;
				GetTopKTwoCluster(image_pair_ids, weights, 
								  clusters[cluster_id1], 
								  clusters[cluster_id2], 
								  image_to_cluster_id_map,
								  image_overlap, edges);
				for (const auto & edge : edges) {
					std::vector<uint32_t> transfer_images1;
					FindTransitivity(transitivity, cluster_id1, 
						edge.first, image_to_cluster_id_map, 
						neighbors, transfer_images1);
					overlaps[cluster_id2].insert(transfer_images1.begin(),
						transfer_images1.end());

					std::vector<uint32_t> transfer_images2;
					FindTransitivity(transitivity, cluster_id2, 
						edge.second, image_to_cluster_id_map, 
						neighbors, transfer_images2);
					overlaps[cluster_id1].insert(transfer_images2.begin(),
						transfer_images2.end());
				} // for edges

				// detect overlap according to sequential charactistic.
				GetSeqOverlapTwoClusters(image_pair_ids, weights,
										 clusters[cluster_id1], 
										 clusters[cluster_id2], 
										 image_to_cluster_id_map,
										 image_dist_seq_overlap, edges);
				for (const auto & edge : edges) {
					std::vector<uint32_t> transfer_images1;
					FindTransitivity(transitivity, cluster_id1, 
						edge.first, image_to_cluster_id_map, 
						neighbors, transfer_images1, image_dist_seq_overlap);
					overlaps[cluster_id2].insert(transfer_images1.begin(),
						transfer_images1.end());

					std::vector<uint32_t> transfer_images2;
					FindTransitivity(transitivity, cluster_id2, 
						edge.second, image_to_cluster_id_map, 
						neighbors, transfer_images2, image_dist_seq_overlap);
					overlaps[cluster_id1].insert(transfer_images2.begin(),
						transfer_images2.end());
				} // for edges
				// for (const auto & edge : edges) {
				// 	overlaps[cluster_id2].insert(edge.first);
				// 	overlaps[cluster_id1].insert(edge.second);
				// }
			} // if
		} // for i
	} // while Q

	// Add overlap images.
	for (int i = 0; i < clusters.size(); ++i) {
		std::cout << "Cluster#" << i << ", overlap image = " 
				  << overlaps[i].size() << std::endl;
		clusters[i]->image_ids.insert(clusters[i]->image_ids.end(), 
			overlaps[i].begin(), overlaps[i].end());
	}
}

std::list<CommunityTree<uint32_t> *> MergeCommunityTree(
	CommunityTree<uint32_t> *root,
	const sensemap::SceneClustering::Options& options){

    std::list<CommunityTree<uint32_t> *> cluster_trees = root->GetLeaves();

    std::cout<<std::endl;
    int num = 0;
    for(auto & cluster_tree : cluster_trees) {
        num++;
        std::cout << cluster_tree->Id() << ": "
                  << cluster_tree->elements().size() << std::endl;
    }
    std::cout<<num<<" clusters before merging"<<std::endl;

    for(auto it = cluster_trees.begin(); it != cluster_trees.end(); ++it) {
        if ((*it)->elements().size() >= options.min_modularity_count){
            continue;
        }
        if ((*it)->isMerged()){ //already merged in early iteration
            it = cluster_trees.erase(it);
            continue;
        }

        auto parent = (*it)->parenet_tree();
        CHECK(parent->child_trees().size() > 1);

        if(parent->elements().size() > options.max_modularity_count) // cannot merge
            continue;

        for(auto &brother : parent->child_trees()){
            brother->setMerged(true);
        }

        it = cluster_trees.erase(it);

        if(parent->elements().size() >= options.min_modularity_count) {
            cluster_trees.push_front(parent);
        } else { // still need merge
            cluster_trees.push_back(parent);
        }
    }

    int invalid_num = 0;
    num = 0;
    std::cout<<std::endl;
    for(auto it = cluster_trees.begin(); it != cluster_trees.end(); )
    {
        if ((*it)->isMerged())
        { //has been merged in later iteration
            it = cluster_trees.erase(it);
            continue;
        }
        if ((*it)->elements().size() < options.min_modularity_count
                || (*it)->elements().size() >= options.max_modularity_count
            ){
            invalid_num++;
        }
        num++;
        std::cout << (*it)->Id() << ": " << (*it)->elements().size()
                  << std::endl;
        ++it;
    }
    std::list<CommunityTree<uint32_t> *>(cluster_trees).swap(cluster_trees);
    std::cout<<num<<" clusters after merging  ("
             << invalid_num << " clusters not satisfied ["
             << options.min_modularity_count <<", " << options.max_modularity_count
             << "] )" << std::endl;
    return cluster_trees;
}

void FastBuildCommunityTree(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	CommunityTree<uint32_t> *community_tree,
	const sensemap::SceneClustering::Options& options) {

	std::vector<Cluster*> clusters;
	FastCommunity(image_pair_ids, clusters, options);
	auto ConstructImagePairs = [&](
			const std::vector<uint32_t>& sub_image_ids,
			std::vector<std::tuple<uint32_t, uint32_t, double> >& sub_image_pair_ids) {
		std::unordered_set<uint32_t> sub_image_ids_set;
		sub_image_ids_set.insert(sub_image_ids.begin(), sub_image_ids.end());
		for (const auto & pair_id : image_pair_ids) {
			if (sub_image_ids_set.count(std::get<0>(pair_id)) &&
			    sub_image_ids_set.count(std::get<1>(pair_id))) {
				sub_image_pair_ids.emplace_back(pair_id);
			}
		}
	};

	if (clusters.size() == 1) {
		if(community_tree->elements().empty()){
			community_tree->SetElements(clusters[0]->image_ids);
		}
		return;
	}

	if(community_tree->elements().empty()) {
		for (const auto & cluster : clusters) {
			community_tree->AddElements(cluster->image_ids);
		}
	}

	for (const auto & cluster : clusters) {
		auto node = new CommunityTree<uint32_t>(cluster->image_ids, community_tree);
		community_tree->addChildTree(node);
		std::cout<<"child: "<< community_tree->child_trees().size()<<std::endl;
		if (cluster->image_ids.size() > options.min_modularity_count) {
			std::vector<std::tuple<uint32_t, uint32_t, double> > image_pair_ids_;
			ConstructImagePairs(cluster->image_ids, image_pair_ids_);
			FastBuildCommunityTree(image_pair_ids_, node, options);
		}
	}
}

void FastCommunityRecurisive(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options) {

	std::vector<Cluster*> clusters_;
	FastCommunity(image_pair_ids, clusters_, options);
	
	auto ConstructImagePairs = [&](
		const std::vector<uint32_t>& sub_image_ids,
		std::vector<std::tuple<uint32_t, uint32_t, double> >& sub_image_pair_ids) {
		std::unordered_set<uint32_t> sub_image_ids_set;
		sub_image_ids_set.insert(sub_image_ids.begin(), sub_image_ids.end());
		for (const auto & pair_id : image_pair_ids) {
			if (sub_image_ids_set.count(std::get<0>(pair_id)) &&
				sub_image_ids_set.count(std::get<1>(pair_id))) {
				sub_image_pair_ids.emplace_back(pair_id);
			}
		}
	};

	if (clusters_.size() > 1) {
		for (const auto & cluster : clusters_) {
			if (cluster->image_ids.size() <= options.max_modularity_count) {
				clusters.emplace_back(cluster);
			} else {
				std::vector<std::tuple<uint32_t, uint32_t, double> > image_pair_ids_;
				ConstructImagePairs(cluster->image_ids, image_pair_ids_);
				FastCommunityRecurisive(image_pair_ids_, clusters, options);
			}
		}
	} else {
		clusters.insert(clusters.end(), clusters_.begin(), clusters_.end());
	}
	return ;
}

// #define DEBUG_INFO

void FastCommunity(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids,
	std::vector<Cluster*> &clusters,
	const sensemap::SceneClustering::Options& options) {
    // default values for parameters which may be modified from the commandline
	ioparm.timer     = 20;
	ioparm.fileFlag  = NONE;
	ioparm.suppFlag  = false;
	ioparm.textFlag  = 0;
	ioparm.filename  = "community.pairs";
	ioparm.s_label   = "a";
	time_t t1;	t1 = time(&t1);
	time_t t2;	t2 = time(&t2);
	
	// ----------------------------------------------------------------------
	// Parse the command line, build filenames and then import the .pairs file
	cout << "\nFast Community Inference.\n";
	cout << "Copyright (c) 2004 by Aaron Clauset (aaron@cs.unm.edu)\n";
	// if (parseCommandLine(argc, argv)) {} else { return 0; }
	cout << "\nimporting: " << ioparm.filename << endl;    // note the input filename
#ifdef DEBUG_INFO
	buildFilenames();								// builds filename strings
#endif
	// readInputFile();								// gets adjacency matrix data
	BuildAdjacentMatrix(image_pair_ids);
	
	// ----------------------------------------------------------------------
	// Allocate data structures for main loop
	a     = new double [gparm.maxid];
	Q     = new double [gparm.n+1];
	joins = new apair  [gparm.n+1];
	for (int i=0; i<gparm.maxid; i++) { a[i] = 0.0; }
	for (int i=0; i<gparm.n+1;   i++) { Q[i] = 0.0; joins[i].x = 0; joins[i].y = 0; }
	int t = 1;
	Qmax.y = -4294967296.0;  Qmax.x = 0;
	if (ioparm.cutstep > 0) { groupListsSetup(); }		// will need to track agglomerations
	
	cout << "now building initial dQ[]" << endl;
	buildDeltaQMatrix();							// builds dQ[] and h
	
	// // initialize f_joins, f_support files
	// ofstream fjoins(ioparm.f_joins.c_str(), ios::trunc);
	// fjoins << -1 << "\t" << -1 << "\t" << Q[0] << "\t0\n";
	// fjoins.close();

#ifdef DEBUG_INFO
	if (ioparm.suppFlag) {
		ofstream fsupp(ioparm.f_support.c_str(), ios::trunc);
		dqSupport();
		fsupp << 0 << "\t" << supportTot << "\t" << supportAve << "\t" << 0 << "\t->\t" << 0 << "\n";
		fsupp.close();
	}
#endif
	
	// ----------------------------------------------------------------------
	// Start FastCommunity algorithm
	cout << "starting algorithm now." << endl;
	tuple  dQmax, dQnew;
	int isupport, jsupport;

//	std::ofstream file("./graph.dot", std::ios::app);
//	file << "digraph G { //" << gparm.n <<std::endl;
//	file << "label = \"community\";" <<std::endl;
	while (h->heapSize() > 1) {
		
		// ---------------------------------
		// Find largest dQ
		if (ioparm.textFlag > 0) { h->printHeapTop10(); cout << endl; }
		dQmax = h->popMaximum();					// select maximum dQ_ij // convention: insert i into j
		if (dQmax.m < -4000000000.0) { break; }		// no more joins possible
		cout << "Q["<<t-1<<"] = "<<Q[t-1];
		
		// ---------------------------------
		// Merge the chosen communities
		cout << "\tdQ = " << dQmax.m << "\t  |H| = " << h->heapSize() << "\n";
		if (dq[dQmax.i].v == NULL || dq[dQmax.j].v == NULL) {
			cout << "WARNING: invalid join (" << dQmax.i << " " << dQmax.j << ") found at top of heap\n"; cin >> pauseme;
		}
		isupport = dq[dQmax.i].v->returnNodecount();
		jsupport = dq[dQmax.j].v->returnNodecount();
		if (isupport < jsupport) {
//			file <<dQmax.i<<" -> "<<dQmax.j
//			     <<"[color=\"0.600 0.999 0.999\"];   ";
//			std::cout<<dQmax.i<<" -> "<<dQmax.j
//			         <<"[color=\"0.600 0.999 0.999\"];"<<std::endl;
			cout << "  join: " << dQmax.i << " -> " << dQmax.j << "\t";
			cout << "(" << isupport << " -> " << jsupport << ")\n";
			mergeCommunities(dQmax.i, dQmax.j);	// merge community i into community j
			joins[t].x = dQmax.i;				// record merge of i(x) into j(y)
			joins[t].y = dQmax.j;				// 
		} else {
//			file<<dQmax.j<<" -> "<<dQmax.i
//			    <<"[color=\"0.600 0.999 0.999\"];  ";
//			std::cout<<dQmax.j<<" -> "<<dQmax.i
//			         <<"[color=\"0.600 0.999 0.999\"];"<<std::endl;
			cout << "  join: " << dQmax.i << " <- " << dQmax.j << "\t";
			cout << "(" << isupport << " <- " << jsupport << ")\n";
			dq[dQmax.i].heap_ptr = dq[dQmax.j].heap_ptr; // take community j's heap pointer
			dq[dQmax.i].heap_ptr->i = dQmax.i;			//   mark it as i's
			dq[dQmax.i].heap_ptr->j = dQmax.j;			//   mark it as i's
			mergeCommunities(dQmax.j, dQmax.i);	// merge community j into community i
			joins[t].x = dQmax.j;				// record merge of j(x) into i(y)
			joins[t].y = dQmax.i;				// 
		}									// 
		Q[t] = dQmax.m + Q[t-1];					// record Q(t)
		
#ifdef DEBUG_INFO
		// ---------------------------------
		// Record join to file
		ofstream fjoins(ioparm.f_joins.c_str(), ios::app);   // open file for writing the next join
		fjoins << joins[t].x-1 << "\t" << joins[t].y-1 << "\t";	// convert to external format
		if ((Q[t] > 0.0 && Q[t] < 0.0000000000001) || (Q[t] < 0.0 && Q[t] > -0.0000000000001))
			{ fjoins << 0.0; } else { fjoins << Q[t]; }
		fjoins << "\t" << t << "\n";
		fjoins.close();
		// Note that it is the .joins file which contains both the dendrogram and the corresponding
		// Q values. The file format is tab-delimited columns of data, where the columns are:
#endif
		// 1. the community which grows
		// 2. the community which was absorbed
		// 3. the modularity value Q after the join
		// 4. the time step value
		
		// ---------------------------------
		// If cutstep valid, then do some work
		if (t <= ioparm.cutstep) { groupListsUpdate(joins[t].x, joins[t].y); }
		if (t == ioparm.cutstep) { recordNetwork(); recordGroupLists(); groupListsStats(); }

#ifdef DEBUG_INFO
		// ---------------------------------
		// Record the support data to file
		if (ioparm.suppFlag) {
			dqSupport();
			ofstream fsupp(ioparm.f_support.c_str(), ios::app);
			// time   remaining support   mean support   support_i --   support_j
			fsupp << t << "\t" << supportTot << "\t" << supportAve << "\t" << isupport;
			if (isupport < jsupport) { fsupp  << "\t->\t"; }
			else { fsupp << "\t<-\t"; }
			fsupp << jsupport << "\n";
			fsupp.close();
		}
#endif
		if (Q[t] > Qmax.y) { Qmax.y = Q[t]; Qmax.x = t; }
		
		t++;									// increment time
	} // ------------- end community merging loop
	//cout << "Q["<<t-1<<"] = "<<Q[t-1] << endl;

#ifdef DEBUG_INFO
	// ----------------------------------------------------------------------
	// Record some results
	t1 = time(&t1);
	ofstream fout(ioparm.f_parm.c_str(), ios::app);
	fout << "---MODULARITY---\n";
	fout << "MAXQ------:\t" << Qmax.y  << "\n";
	fout << "STEP------:\t" << Qmax.x  << "\n";
	fout << "EXIT------:\t" << asctime(localtime(&t1));
	fout.close();
#endif
	// BuildLabels(t, 2, clusters);

	BuildLabels(t - 1, clusters, options);
//	file << std::endl;
//	file << "}" <<std::endl;
	// MergeClusters(image_pair_ids, clusters);

	cout << "exited safely" << endl;
}

void BuildAdjacentMatrix(
	const std::vector<std::tuple<uint32_t, uint32_t, double> > &image_pair_ids) {
	// temporary variables for this function
	int numnodes = 0;
	int numlinks = 0;
	double sumwieght = 0;
	int s,f,t;
	double w;
	edge **last;
	edge *newedge;
	edge *current;								// pointer for checking edge existence
	bool existsFlag;							// flag for edge existence
	time_t t1; t1 = time(&t1);
	time_t t2; t2 = time(&t2);
	
	// First scan through the input file to discover the largest node id. We need to
	// do this so that we can allocate a properly sized array for the sparse matrix
	// representation.
	cout << " scanning input file for basic information." << endl;
	cout << "  edgecount: [0]"<<endl;

	for (const auto & image_pair_id : image_pair_ids) {
		s = std::get<0>(image_pair_id);
		f = std::get<1>(image_pair_id);
		numlinks++;								// count number of edges
		if (f < s) { t = s; s = f; f = t; }		// guarantee s < f
		if (f > numnodes) { numnodes = f; }		// track largest node index

		if (t2-t1>ioparm.timer) {				// check timer; if necessarsy, display
			cout << "  edgecount: ["<<numlinks<<"]"<<endl;
			t1 = t2; 
			ioparm.timerFlag = true; 
		} 
		t2=time(&t2);
	}

	cout << "  edgecount: ["<<numlinks<<"] total (first pass)"<<endl;
	gparm.maxid = numnodes+2;					// store maximum index
	elist = new edge [2*numlinks];				// create requisite number of edges
	int ecounter = 0;							// index of next edge of elist to be used

	// Now that we know numnodes, we can allocate the space for the sparse matrix, and
	// then reparse the file, adding edges as necessary.
	cout << " allocating space for network." << endl;
	e        = new  edge [gparm.maxid];			// (unordered) sparse adjacency matrix
	last     = new edge* [gparm.maxid];			// list of pointers to the last edge in each row
	numnodes = 0;								// numnodes now counts number of actual used node ids
	numlinks = 0;								// numlinks now counts number of bi-directional edges created
	sumwieght = 0;
	ioparm.timerFlag = false;					// reset timer
	
#ifdef DEBUG_INFO
	cout << " reparsing the input file to build network data structure." << endl;
	cout << "  edgecount: [0]"<<endl;
	ifstream fin(ioparm.f_input.c_str(), ios::in);
#endif
	for (const auto & image_pair_id : image_pair_ids) {	
		s = std::get<0>(image_pair_id);
		f = std::get<1>(image_pair_id);
		w = std::get<2>(image_pair_id);
		// s++; f++;								// increment s,f to prevent using e[0]
		if (f < s) { t = s; s = f; f = t; }		// guarantee s < f
		numlinks++;							// increment link count (preemptive)
		sumwieght += w;
		if (e[s].so == 0) {						// if first edge with s, add s and (s,f)
			e[s].so = s;						// 
			e[s].si = f;						//
			e[s].weight = w;
			last[s] = &e[s];					//    point last[s] at self
			numnodes++;						//    increment node count
		} else {								//    try to add (s,f) to s-edgelist
			current = &e[s];					// 
			existsFlag = false;					// 
			while (current != NULL) {			// check if (s,f) already in edgelist
				if (current->si==f) {			// 
					existsFlag = true;			//    link already exists
					numlinks--;				//    adjust link-count downward
					sumwieght -= w;
					break;					// 
				}							// 
				current = current->next;			//    look at next edge
			}								// 
			if (!existsFlag) {					// if not already exists, append it
				newedge = &elist[ecounter++];		//    grab next-free-edge
				newedge -> so = s;				// 
				newedge -> si = f;				//
				newedge -> weight = w;
				last[s] -> next = newedge;		//    append newedge to [s]'s list
				last[s]         = newedge;		//    point last[s] to newedge
			}								// 
		}									// 
		
		if (e[f].so == 0) {						// if first edge with f, add f and (f,s)
			e[f].so = f;						// 
			e[f].si = s;						//
			e[f].weight = w;
			last[f] = &e[f];					//    point last[s] at self
			numnodes++;						//    increment node count
		} else {								// try to add (f,s) to f-edgelist
			if (!existsFlag) {					//    if (s,f) wasn't in s-edgelist, then
				newedge = &elist[ecounter++];		//       (f,s) not in f-edgelist
				newedge -> so = f;				// 
				newedge -> si = s;				//
				newedge -> weight = w;
				last[f] -> next = newedge;		//    append newedge to [f]'s list
				last[f]		 = newedge;		//    point last[f] to newedge
			}								// 
		}									
		existsFlag = false;						// reset existsFlag
		if (t2-t1>ioparm.timer) {				// check timer; if necessarsy, display
			cout << "  edgecount: ["<<numlinks<<"]"<<endl;
			t1 = t2;							// 
			ioparm.timerFlag = true;				// 
		}									// 
		t2=time(&t2);							// 
		
	}
	cout << "  edgecount: ["<<numlinks<<"] total (second pass)"<<endl;
#ifdef DEBUG_INFO
	fin.close();
	
	// Now we record our work in the parameters file, and exit.
	ofstream fout(ioparm.f_parm.c_str(), ios::app);
	fout << "---NET_STATS----\n";
	fout << "MAXID-----:\t" << gparm.maxid-2 << "\n";
	fout << "NUMNODES--:\t" << numnodes << "\n";
	fout << "NUMEDGES--:\t" << numlinks << "\n";
	fout << "NUMW--:\t" << sumwieght << "\n";
	fout.close();
#endif

	gparm.m = numlinks;							// store actual number of edges created
	gparm.n = numnodes;							// store actual number of nodes used
	gparm.w = sumwieght;
	return;
}

void buildDeltaQMatrix() {
	
	// Given that we've now populated a sparse (unordered) adjacency matrix e (e), 
	// we now need to construct the intial dQ matrix according to the definition of dQ
	// which may be derived from the definition of modularity Q:
	//    Q(t) = \sum_{i} (e_{ii} - a_{i}^2) = Tr(e) - ||e^2||
	// thus dQ is
	//    dQ_{i,j} = 2* ( e_{i,j} - a_{i}a_{j} )
	//    where a_{i} = \sum_{j} e_{i,j} (i.e., the sum over the ith row)
	// To create dQ, we must insert each value of dQ_{i,j} into a binary search tree,
	// for the jth column. That is, dQ is simply an array of such binary search trees,
	// each of which represents the dQ_{x,j} adjacency vector. Having created dQ as
	// such, we may happily delete the matrix e in order to free up more memory.
	// The next step is to create a max-heap data structure, which contains the entries
	// of the following form (value, s, t), where the heap-key is 'value'. Accessing the
	// root of the heap gives us the next dQ value, and the indices (s,t) of the vectors
	// in dQ which need to be updated as a result of the merge.
	
	
	// First we compute e_{i,j}, and the compute+store the a_{i} values. These will be used
	// shortly when we compute each dQ_{i,j}.
	edge   *current;
	double  eij = (double)(0.5/gparm.m);				// intially each e_{i,j} = 1/m
	for (int i=1; i<gparm.maxid; i++) {				// for each row
		a[i] = 0.0;								// 
		if (e[i].so != 0) {							//    ensure it exists
			current = &e[i];//    grab first edge

			a[i] = current->weight * 0.5 / gparm.w;							//eij   initialize a[i]

			// std::cout << std::fixed << std::setprecision(6) <<eij<<" "<<current->weight * 0.5 / gparm.w<<std::endl;

			while (current->next != NULL) {			//    loop through remaining edges
				a[i] += current->next->weight * 0.5 / gparm.w;		//eij       add another eij

				current = current->next;				//
			}
			Q[0] += -1.0*a[i]*a[i];					// calculate initial value of Q
		}
	}

	// now we create an empty (ordered) sparse matrix dq[]
	dq = new nodenub [gparm.maxid];						// initialize dq matrix
	for (int i=0; i<gparm.maxid; i++) {					// 
		dq[i].heap_ptr = NULL;							// no pointer in the heap at first
		if (e[i].so != 0) { dq[i].v = new vektor(2+(int)floor(gparm.m*a[i])); }
		else {			dq[i].v = NULL; }
	}
	h = new maxheap(gparm.n);						// allocate max-heap of size = number of nodes
	
	// Now we do all the work, which happens as we compute and insert each dQ_{i,j} into 
	// the corresponding (ordered) sparse vector dq[i]. While computing each dQ for a
	// row i, we track the maximum dQmax of the row and its (row,col) indices (i,j). Upon
	// finishing all dQ's for a row, we insert the tuple into the max-heap hQmax. That
	// insertion returns the itemaddress, which we then store in the nodenub heap_ptr for 
	// that row's vector.
	double    dQ;
	tuple	dQmax;										// for heaping the row maxes
	tuple*    itemaddress;									// stores address of item in maxheap

	for (int i=1; i<gparm.maxid; i++) {
		if (e[i].so != 0) {
			current = &e[i];								// grab first edge
			dQ      = 2.0*(current->weight * 0.5 / gparm.w -
					(a[current->so]*a[current->si]));   //eij compute its dQ
			dQmax.m = dQ;									// assume it is maximum so far
			dQmax.i = current->so;							// store its (row,col)
			dQmax.j = current->si;							// 
			dq[i].v->insertItem(current->si, dQ);				// insert its dQ
			while (current->next != NULL) {					// 
				current = current->next;						// step to next edge
				dQ = 2.0*(current->weight * 0.5 / gparm.w -
						(a[current->so]*a[current->si]));	//eij compute new dQ
				if (dQ > dQmax.m) {							// if dQ larger than current max
					dQmax.m = dQ;							//    replace it as maximum so far
					dQmax.j = current->si;					//    and store its (col)
				}
				dq[i].v->insertItem(current->si, dQ);			// insert it into vector[i]
			}
			dq[i].heap_ptr = h->insertItem(dQmax);				// store the pointer to its loc in heap
		}
	}

	delete [] elist;								// free-up adjacency matrix memory in two shots
	delete [] e;									// 
	return;
}

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

void buildFilenames() {

	ioparm.f_input   = ioparm.d_in  + ioparm.filename;
	ioparm.f_parm    = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".info";
	ioparm.f_joins   = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".joins";
	ioparm.f_support = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".supp";
	ioparm.f_net     = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".wpairs";
	ioparm.f_group   = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".groups";
	ioparm.f_gstats  = ioparm.d_out + ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".hist";
	
	if (true) { ofstream flog(ioparm.f_parm.c_str(), ios::trunc); flog.close(); }
	time_t t; t = time(&t);
	ofstream flog(ioparm.f_parm.c_str(), ios::app);
	flog << "FASTCOMMUNITY_INFERENCE_ALGORITHM\n";
	flog << "START-----:\t" << asctime(localtime(&t));
	flog << "---FILES--------\n";
	flog << "DIRECTORY-:\t" << ioparm.d_out		<< "\n";
	flog << "F_IN------:\t" << ioparm.filename   << "\n";
	flog << "F_JOINS---:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".joins" << "\n";
	flog << "F_INFO----:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".info"  << "\n";
	if (ioparm.suppFlag) {
		flog << "F_SUPP----:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".supp" << "\n"; }
	if (ioparm.cutstep>0) {
		flog << "F_NET-----:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".wpairs" << "\n";
		flog << "F_GROUPS--:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".groups" << "\n";
		flog << "F_GDIST---:\t" << ioparm.s_scratch + "-fc_"  + ioparm.s_label + ".hist"   << "\n";
	}
	flog.close();
	
	return;
}

//----------------------------------------------------------------------------//
// returns the support of the dQ[]
//----------------------------------------------------------------------------//

void dqSupport() {
	int    total = 0;
	int    count = 0;
	for (int i=0; i<gparm.maxid; i++) {
		if (dq[i].heap_ptr != NULL) { total += dq[i].v->returnNodecount(); count++; }
	}
	supportTot = total;
	supportAve = total/(double)count;
	return;
}

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

void groupListsSetup() {
	
	list *newList;
	c = new stub [gparm.maxid];
	for (int i=0; i<gparm.maxid; i++) {
		if (e[i].so != 0) {								// note: internal indexing
			newList = new list;							// create new community member
			newList->index = i;							//    with index i
			c[i].members   = newList;					// point ith community at newList
			c[i].size		= 1;							// point ith community at newList
			c[i].last		= newList;					// point last[] at that element too
			c[i].valid	= true;						// mark as valid community
		}
	}
	
	return;
}

//----------------------------------------------------------------------------//
// function for computing statistics on the list of groups
//----------------------------------------------------------------------------//

void groupListsStats() {

	gstats.numgroups = 0;
	gstats.maxsize   = 0;
	gstats.minsize   = gparm.maxid;
	double count     = 0.0;
	for (int i=0; i<gparm.maxid; i++) {
		if (c[i].valid) {
			gstats.numgroups++;							// count number of communities
			count += 1.0;
			if (c[i].size > gstats.maxsize) { gstats.maxsize = c[i].size; }  // find biggest community
			if (c[i].size < gstats.minsize) { gstats.minsize = c[i].size; }  // find smallest community
			// compute mean group size
			gstats.meansize = (double)(c[i].size)/count + (((double)(count-1.0)/count)*gstats.meansize);
		}
	}
	
	count = 0.0;
	gstats.sizehist = new double [gstats.maxsize+1];
	for (int i=0; i<gstats.maxsize+1; i++) { gstats.sizehist[i] = 0; }
	for (int i=0; i<gparm.maxid; i++) {
		if (c[i].valid) {
			gstats.sizehist[c[i].size] += 1.0;					// tabulate histogram of sizes
			count += 1.0;
		}
	}
	// convert histogram to pdf, and write it to disk
	for (int i=0; i<gstats.maxsize+1; i++) { gstats.sizehist[i] = gstats.sizehist[i]/count; }
	ofstream fgstat(ioparm.f_gstats.c_str(), ios::trunc);
	for (int i=gstats.minsize; i<gstats.maxsize+1; i++) {
		fgstat << i << "\t" << gstats.sizehist[i] << "\n";
	}
	fgstat.close();
	
	// record some statistics
	time_t t1; t1 = time(&t1);
	ofstream fout(ioparm.f_parm.c_str(), ios::app);
	fout << "---GROUPS-------\n";
	fout << "NUMGROUPS-:\t" << gstats.numgroups  << "\n";
	fout << "MINSIZE---:\t" << gstats.minsize    << "\n";
	fout << "MEANSIZE--:\t" << gstats.meansize   << "\n";
	fout << "MAXSIZE---:\t" << gstats.maxsize    << "\n";
	fout.close();
	return;
}

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

void groupListsUpdate(const int x, const int y) {
	
	c[y].last->next = c[x].members;				// attach c[y] to end of c[x]
	c[y].last		 = c[x].last;					// update last[] for community y
	c[y].size		 += c[x].size;					// add size of x to size of y
	
	c[x].members   = NULL;						// delete community[x]
	c[x].valid	= false;						// 
	c[x].size		= 0;							// 
	c[x].last		= NULL;						// delete last[] for community x
	
	return;
}

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

void mergeCommunities(int i, int j) {
	
	// To do the join operation for a pair of communities (i,j), we must update the dQ
	// values which correspond to any neighbor of either i or j to reflect the change.
	// In doing this, there are three update rules (cases) to follow:
	//  1. jix-triangle, in which the community x is a neighbor of both i and j
	//  2. jix-chain, in which community x is a neighbor of i but not j
	//  3. ijx-chain, in which community x is a neighbor of j but not i
	//
	// For the first two cases, we may make these updates by simply getting a list of
	// the elements (x,dQ) of [i] and stepping through them. If x==j, then we can ignore
	// that value since it corresponds to an edge which is being absorbed by the joined
	// community (i,j). If [j] also contains an element (x,dQ), then we have a triangle;
	// if it does not, then we have a jix-chain.
	//
	// The last case requires that we step through the elements (x,dQ) of [j] and update each
	// if [i] does not have an element (x,dQ), since that implies a ijx-chain.
	// 
	// Let d([i]) be the degree of the vector [i], and let k = d([i]) + d([j]). The running
	// time of this operation is O(k log k)
	//
	// Essentially, we do most of the following operations for each element of
	// dq[i]_x where x \not= j
	//  1.  add dq[i]_x to dq[j]_x (2 cases)
	//  2.  remove dq[x]_i
	//  3.  update maxheap[x]
	//  4.  add dq[i]_x to dq[x]_j (2 cases)
	//  5.  remove dq[j]_i
	//  6.  update maxheap[j]
	//  7.  update a[j] and a[i]
	//  8.  delete dq[i]
	
	dpair *list, *current, *temp;
	tuple newMax;
	int t = 1;
	
	// -- Working with the community being inserted (dq[i])
	// The first thing we must do is get a list of the elements (x,dQ) in dq[i]. With this 
	// list, we can then insert each into dq[j].

	//	dq[i].v->printTree();
	list    = dq[i].v->returnTreeAsList();			// get a list of items in dq[i].v
	current = list;							// store ptr to head of list
	
	if (ioparm.textFlag>1) {
		cout << "stepping through the "<<dq[i].v->returnNodecount() << " elements of community " << i << endl;
	}
		
	// ---------------------------------------------------------------------------------
	// SEARCHING FOR JIX-TRIANGLES AND JIX-CHAINS --------------------------------------
	// Now that we have a list of the elements of [i], we can step through them to check if
	// they correspond to an jix-triangle, a jix-chain, or the edge (i,j), and do the appropriate
	// operation depending.
	
	while (current!=NULL) {						// insert list elements appropriately
		
		if (ioparm.textFlag>1) { cout << endl << "element["<<t<<"] from dq["<<i<<"] is ("<<current->x<<" "<<current->y<<")" << endl; }

		// If the element (x,dQ) is actually (j,dQ), then we can ignore it, since it will 
		// correspond to an edge internal to the joined community (i,j) after the join.
		if (current->x != j) {

			// Now we must decide if we have a jix-triangle or a jix-chain by asking if
			// [j] contains some element (x,dQ). If the following conditional is TRUE,
			// then we have a jix-triangle, ELSE it is a jix-chain.
			
			if (dq[j].v->findItem(current->x)) {
				// CASE OF JIX-TRIANGLE
				if (ioparm.textFlag>1) {
					cout << "  (0) case of triangle: e_{"<<current->x<<" "<<j<<"} exists" << endl;
					cout << "  (1) adding ("<<current->x<<" "<<current->y<<") to dq["<<current->x<<"] as ("<<j<<" "<<current->y<<")"<<endl;
				}
				
				// We first add (x,dQ) from [i] to [x] as (j,dQ), since [x] essentially now has
				// two connections to the joined community [j].
				if (ioparm.textFlag>1) {
					cout << "  (1) heapsize = " << dq[current->x].v->returnHeaplimit() << endl; 
					cout << "  (1) araysize = " << dq[current->x].v->returnArraysize() << endl; 
					cout << "  (1) vectsize = " << dq[current->x].v->returnNodecount() << endl; }
				dq[current->x].v->insertItem(j,current->y);			// (step 1)

				// Then we need to delete the element (i,dQ) in [x], since [i] is now a
				// part of [j] and [x] must reflect this connectivity.
				if (ioparm.textFlag>1) { cout << "  (2) now we delete items associated with "<< i << " in dq["<<current->x<<"]" << endl; }
				dq[current->x].v->deleteItem(i);					// (step 2)

				// After deleting an item, the tree may now have a new maximum element in [x],
				// so we need to check it against the old maximum element. If it's new, then
				// we need to update that value in the heap and reheapify.
				newMax = dq[current->x].v->returnMaxStored();		// (step 3)
				if (ioparm.textFlag>1) { cout << "  (3) and dq["<<current->x<<"]'s new maximum is (" << newMax.m <<" "<<newMax.j<< ") while the old maximum was (" << dq[current->x].heap_ptr->m <<" "<<dq[current->x].heap_ptr->j<<")"<< endl; }
//				if (newMax.m > dq[current->x].heap_ptr->m || dq[current->x].heap_ptr->j==i) {
					h->updateItem(dq[current->x].heap_ptr, newMax);
					if (ioparm.textFlag>1) { cout << "  updated dq["<<current->x<<"].heap_ptr to be (" << dq[current->x].heap_ptr->m <<" "<<dq[current->x].heap_ptr->j<<")"<< endl; }
//				}
// Change suggested by Janne Aukia (jaukia@cc.hut.fi) on 12 Oct 2006
				
				// Finally, we must insert (x,dQ) into [j] to note that [j] essentially now
				// has two connections with its neighbor [x].
				if (ioparm.textFlag>1) { cout << "  (4) adding ("<<current->x<<" "<<current->y<<") to dq["<<j<<"] as ("<<current->x<<" "<<current->y<<")"<<endl; }
				dq[j].v->insertItem(current->x,current->y);		// (step 4)
				
			} else {
				// CASE OF JIX-CHAIN
				
				// The first thing we need to do is calculate the adjustment factor (+) for updating elements.
				double axaj = -2.0*a[current->x]*a[j];
				if (ioparm.textFlag>1) {
					cout << "  (0) case of jix chain: e_{"<<current->x<<" "<<j<<"} absent" << endl;
					cout << "  (1) adding ("<<current->x<<" "<<current->y<<") to dq["<<current->x<<"] as ("<<j<<" "<<current->y+axaj<<")"<<endl;
				}
				
				// Then we insert a new element (j,dQ+) of [x] to represent that [x] has
				// acquired a connection to [j], which was [x]'d old connection to [i]
				dq[current->x].v->insertItem(j,current->y + axaj);	// (step 1)
				
				// Now the deletion of the connection from [x] to [i], since [i] is now
				// a part of [j]
				if (ioparm.textFlag>1) { cout << "  (2) now we delete items associated with "<< i << " in dq["<<current->x<<"]" << endl; }
				dq[current->x].v->deleteItem(i);					// (step 2)
				
				// Deleting that element may have changed the maximum element for [x], so we
				// need to check if the maximum of [x] is new (checking it against the value
				// in the heap) and then update the maximum in the heap if necessary.
				newMax = dq[current->x].v->returnMaxStored();		// (step 3)
				if (ioparm.textFlag>1) { cout << "  (3) and dq["<<current->x<<"]'s new maximum is (" << newMax.m <<" "<<newMax.j<< ") while the old maximum was (" << dq[current->x].heap_ptr->m <<" "<<dq[current->x].heap_ptr->j<<")"<< endl; }
//				if (newMax.m > dq[current->x].heap_ptr->m || dq[current->x].heap_ptr->j==i) {
					h->updateItem(dq[current->x].heap_ptr, newMax);
					if (ioparm.textFlag>1) { cout << "  updated dq["<<current->x<<"].heap_ptr to be (" << dq[current->x].heap_ptr->m <<" "<<dq[current->x].heap_ptr->j<<")"<< endl; }
//				}
// Change suggested by Janne Aukia (jaukia@cc.hut.fi) on 12 Oct 2006
					
				// Finally, we insert a new element (x,dQ+) of [j] to represent [j]'s new
				// connection to [x]
				if (ioparm.textFlag>1) { cout << "  (4) adding ("<<current->x<<" "<<current->y<<") to dq["<<j<<"] as ("<<current->x<<" "<<current->y+axaj<<")"<<endl; }
				dq[j].v->insertItem(current->x,current->y + axaj);	// (step 4)

			}    // if (dq[j].v->findItem(current->x))
			
		}    // if (current->x != j)
		
		temp    = current;
		current = current->next;						// move to next element
		delete temp;
		temp = NULL;
		t++;
	}    // while (current!=NULL)

	// We've now finished going through all of [i]'s connections, so we need to delete the element
	// of [j] which represented the connection to [i]
	if (ioparm.textFlag>1) {
		cout << endl;
		cout << "  whoops. no more elements for community "<< i << endl;
		cout << "  (5) now deleting items associated with "<<i<< " (the deleted community) in dq["<<j<<"]" << endl;
	}

	if (ioparm.textFlag>1) { dq[j].v->printTree(); }
	dq[j].v->deleteItem(i);						// (step 5)
	
	// We can be fairly certain that the maximum element of [j] was also the maximum
	// element of [i], so we need to check to update the maximum value of [j] that
	// is in the heap.
	newMax = dq[j].v->returnMaxStored();			// (step 6)
	if (ioparm.textFlag>1) { cout << "  (6) dq["<<j<<"]'s old maximum was (" << dq[j].heap_ptr->m <<" "<< dq[j].heap_ptr->j<< ")\t"<<dq[j].heap_ptr<<endl; }
	h->updateItem(dq[j].heap_ptr, newMax);
	if (ioparm.textFlag>1) { cout << "      dq["<<j<<"]'s new maximum  is (" << dq[j].heap_ptr->m <<" "<< dq[j].heap_ptr->j<< ")\t"<<dq[j].heap_ptr<<endl; }
	if (ioparm.textFlag>1) { dq[j].v->printTree(); }
	
	// ---------------------------------------------------------------------------------
	// SEARCHING FOR IJX-CHAINS --------------------------------------------------------
	// So far, we've treated all of [i]'s previous connections, and updated the elements
	// of dQ[] which corresponded to neighbors of [i] (which may also have been neighbors
	// of [j]. Now we need to update the neighbors of [j] (as necessary)
	
	// Again, the first thing we do is get a list of the elements of [j], so that we may
	// step through them and determine if that element constitutes an ijx-chain which
	// would require some action on our part.
	list = dq[j].v->returnTreeAsList();			// get a list of items in dq[j].v
	current = list;							// store ptr to head of list
	t       = 1;
	if (ioparm.textFlag>1) { cout << "\nstepping through the "<<dq[j].v->returnNodecount() << " elements of community " << j << endl; }

	while (current != NULL) {					// insert list elements appropriately
		if (ioparm.textFlag>1) { cout << endl << "element["<<t<<"] from dq["<<j<<"] is ("<<current->x<<" "<<current->y<<")" << endl; }

		// If the element (x,dQ) of [j] is not also (i,dQ) (which it shouldn't be since we've
		// already deleted it previously in this function), and [i] does not also have an
		// element (x,dQ), then we have an ijx-chain.
		if ((current->x != i) && (!dq[i].v->findItem(current->x))) {
			// CASE OF IJX-CHAIN
			
			// First we must calculate the adjustment factor (+).
			double axai = -2.0*a[current->x]*a[i];
			if (ioparm.textFlag>1) {
				cout << "  (0) case of ijx chain: e_{"<<current->x<<" "<<i<<"} absent" << endl;
				cout << "  (1) updating dq["<<current->x<<"] to ("<<j<<" "<<current->y+axai<<")"<<endl;
			}
			
			// Now we must add an element (j,+) to [x], since [x] has essentially now acquired
			// a new connection to [i] (via [j] absorbing [i]).
			dq[current->x].v->insertItem(j, axai);			// (step 1)

			// This new item may have changed the maximum dQ of [x], so we must update it.
			newMax = dq[current->x].v->returnMaxStored();	// (step 3)
			if (ioparm.textFlag>1) { cout << "  (3) dq["<<current->x<<"]'s old maximum was (" << dq[current->x].heap_ptr->m <<" "<< dq[current->x].heap_ptr->j<< ")\t"<<dq[current->x].heap_ptr<<endl; }
			h->updateItem(dq[current->x].heap_ptr, newMax);
			if (ioparm.textFlag>1) {
				cout << "      dq["<<current->x<<"]'s new maximum  is (" << dq[current->x].heap_ptr->m <<" "<< dq[current->x].heap_ptr->j<< ")\t"<<dq[current->x].heap_ptr<<endl;
				cout << "  (4) updating dq["<<j<<"] to ("<<current->x<<" "<<current->y+axai<<")"<<endl;
			}

			// And we must add an element (x,+) to [j], since [j] as acquired a new connection
			// to [x] (via absorbing [i]).
			dq[j].v->insertItem(current->x, axai);			// (step 4)
			newMax = dq[j].v->returnMaxStored();			// (step 6)
			if (ioparm.textFlag>1) { cout << "  (6) dq["<<j<<"]'s old maximum was (" << dq[j].heap_ptr->m <<" "<< dq[j].heap_ptr->j<< ")\t"<<dq[j].heap_ptr<<endl; }
			h->updateItem(dq[j].heap_ptr, newMax);
			if (ioparm.textFlag>1) { cout << "      dq["<<j<<"]'s new maximum  is (" << dq[j].heap_ptr->m <<" "<< dq[j].heap_ptr->j<< ")\t"<<dq[j].heap_ptr<<endl; }

		}    //  (current->x != i && !dq[i].v->findItem(current->x))
		
		temp    = current;
		current = current->next;						// move to next element
		delete temp;
		temp = NULL;
		t++;
	}    // while (current!=NULL)
	
	// Now that we've updated the connections values for all of [i]'s and [j]'s neighbors, 
	// we need to update the a[] vector to reflect the change in fractions of edges after
	// the join operation.
	if (ioparm.textFlag>1) {
		cout << endl;
		cout << "  whoops. no more elements for community "<< j << endl;
		cout << "  (7) updating a["<<j<<"] = " << a[i] + a[j] << " and zeroing out a["<<i<<"]" << endl;
	}
	a[j] += a[i];								// (step 7)
	a[i] = 0.0;
	
	// ---------------------------------------------------------------------------------
	// Finally, now we need to clean up by deleting the vector [i] since we'll never
	// need it again, and it'll conserve memory. For safety, we also set the pointers
	// to be NULL to prevent inadvertent access to the deleted data later on.

	if (ioparm.textFlag>1) { cout << "--> finished merging community "<<i<<" into community "<<j<<" and housekeeping.\n\n"; }
	delete dq[i].v;							// (step 8)
	dq[i].v        = NULL;						// (step 8)
	dq[i].heap_ptr = NULL;						//
	
	return;
 
}

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

bool parseCommandLine(int argc,char * argv[]) {
	int argct = 1;
	string temp, ext;
	string::size_type pos;
	char **endptr;
	long along;
	int count;

	// Description of commandline arguments
	// -f <filename>    give the target .pairs file to be processed
	// -l <text>		the text label for this run; used to build output filenames
	// -t <int>		period of timer for reporting progress of computation to screen
	// -s			calculate and track the support of the dQ matrix
	// -v --v ---v		differing levels of screen output verbosity
	// -c <int>		record the aglomerated network at step <int>
	
	if (argc <= 1) { // if no arguments, return statement about program usage.
		cout << "\nThis program runs the fast community structure inference algorithm due to ";
		cout << "Clauset, Newman and Moore on an input graph in the .pairs format. This version ";
		cout << "is the full max-heap version originally described in cond-mat/0408187. The program ";
		cout << "requires the input network connectivity to be formatted in the following specific ";
		cout << "way: the graph must be simple and connected, where each edge is written on ";
		cout << "a line in the format 'u v' (e.g., 3481 3483).\n";
		cout << "To run the program, you must specify the input network file (-f file.pairs). ";
		cout << "Additionally, you can differentiate runs on the same input file with a label ";
		cout << "(-l test_run) which is imbedded in all corresponding output files. ";
		cout << "Because the algorithm is deterministic, you can specify a point (-c C) at which to ";
		cout << "cut the dendrogram; the program will write out various information about the clustered ";
		cout << "network: a list of clusters, the clustered connectivity, and the cluster size ";
		cout << "distribution. Typically, one wants this value to be the time at which modularity Q ";
		cout << "was maximized (that time is recorded in the .info file for a given run).\n";
		cout << "Examples:\n";
		cout << "  ./FastCommunity -f network.pairs -l test_run\n";
		cout << "  ./FastCommunity -f network.pairs -l test_run -c 1232997\n";
		cout << "\n";
		return false;
	}

	while (argct < argc) {
		temp = argv[argct];
		
		if (temp == "-files") {
			cout << "\nBasic files generated:\n";
			cout << "-- .INFO\n";
			cout << "   Various information about the program's running. Includes a listing of ";
			cout << "the files it generates, number of vertices and edges processed, the maximum ";
			cout << "modularity found and the corresponding step (you can re-run the program with ";
			cout << "this value in the -c argument to have it output the contents of the clusters, ";
			cout << "etc. when it reaches that step again (not the most efficient solution, but it ";
			cout << "works)), start/stop time, and when -c is used, it records some information about ";
			cout << "the distribution of cluster sizes.\n";
			cout << "-- .JOINS\n";
			cout << "   The dendrogram and modularity information from the algorithm. The file format ";
			cout << "is tab-delimited columns of data, where the columns are:\n";
			cout << " 1. the community index which absorbs\n";
			cout << " 2. the community index which was absorbed\n";
			cout << " 3. the modularity value Q after the join\n";
			cout << " 4. the time step of the join\n";
			cout << "\nOptional files generated (at time t=C when -c C argument used):\n";
			cout << "-- .WPAIRS\n";
			cout << "   The connectivity of the clustered graph in a .wpairs file format ";
			cout << "(i.e., weighted edges). The edge weights should be the dQ values associated ";
			cout << "with that clustered edge at time C. From this format, it's easy to ";
			cout << "convert into another for visualization (e.g., pajek's .net format).\n";
			cout << "-- .HIST\n";
			cout << "   The size distribution of the clusters.\n";
			cout << "-- .GROUPS\n";
			cout << "   A list of each group and the names of the vertices which compose it (this is ";
			cout << "particularly useful for verifying that the clustering makes sense - tedious but ";
			cout << "important).\n";
			cout << "\n";
			return false;
		} else if (temp == "-f") {			// input file name
			argct++;
			temp = argv[argct];
			ext = ".pairs";
			pos = temp.find(ext,0);
			if (pos == string::npos) { cout << " Error: Input file must have terminating .pairs extension.\n"; return false; }
			ext = "/";
			count = 0; pos = string::npos;
			for (int i=0; i < temp.size(); i++) { if (temp[i] == '/') { pos = i; } }
			if (pos == string::npos) {
				ioparm.d_in = "";
				ioparm.filename = temp;
			} else {
				ioparm.d_in = temp.substr(0, pos+1);
				ioparm.filename = temp.substr(pos+1,temp.size()-pos-1);
			}
			ioparm.d_out = ioparm.d_in;
			// now grab the filename sans extension for building outputs files
			for (int i=0; i < ioparm.filename.size(); i++) { if (ioparm.filename[i] == '.') { pos = i; } }
			ioparm.s_scratch = ioparm.filename.substr(0,pos);
		} else if (temp == "-l") {	// s_label
			argct++;
			if (argct < argc) { ioparm.s_label = argv[argct]; }
			else { " Warning: missing modifier for -l argument; using default.\n"; }
			
		} else if (temp == "-t") {	// timer value
			argct++;
			if (argct < argc) {
				along = strtol(argv[argct],endptr,10);
				ioparm.timer = atoi(argv[argct]);
				cout << ioparm.timer << endl;
				if (ioparm.timer == 0 || std::string(argv[argct]).size() > temp.length()) {
					cout << " Warning: malformed modifier for -t; using default.\n"; argct--;
					ioparm.timer = 20;
				} 
			} else {
				cout << " Warning: missing modifier for -t argument; using default.\n"; argct--;
			}
		} else if (temp == "-c") {	// cut value
			argct++;
			if (argct < argc) {
//				along = strtol(argv[argct],endptr,10);
				ioparm.cutstep = atoi(argv[argct]);
				if (ioparm.cutstep == 0) {
					cout << " Warning: malformed modifier for -c; disabling output.\n"; argct--;
				} 
			} else {
				cout << " Warning: missing modifier for -t argument; using default.\n"; argct--;
			}
		}
		else if (temp == "-s")		{    ioparm.suppFlag = true;		}
		else if (temp == "-v")		{    ioparm.textFlag = 1;		}
		else if (temp == "--v")		{    ioparm.textFlag = 2;		}
		else if (temp == "---v")		{    ioparm.textFlag = 3;		}
		else {  cout << "Unknown commandline argument: " << argv[argct] << endl; }
		argct++;
	}
		
	return true;
}

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

void readInputFile() {
	
	// temporary variables for this function
	int numnodes = 0;
	int numlinks = 0;
	int s,f,t;
	edge **last;
	edge *newedge;
	edge *current;								// pointer for checking edge existence
	bool existsFlag;							// flag for edge existence
	time_t t1; t1 = time(&t1);
	time_t t2; t2 = time(&t2);
	
	// First scan through the input file to discover the largest node id. We need to
	// do this so that we can allocate a properly sized array for the sparse matrix
	// representation.
	cout << " scanning input file for basic information." << endl;
	cout << "  edgecount: [0]"<<endl;
	ifstream fscan(ioparm.f_input.c_str(), ios::in);
	while (fscan >> s >> f) {					// read friendship pair (s,f)
		numlinks++;							// count number of edges
		if (f < s) { t = s; s = f; f = t; }		// guarantee s < f
		if (f > numnodes) { numnodes = f; }		// track largest node index

		if (t2-t1>ioparm.timer) {				// check timer; if necessarsy, display
			cout << "  edgecount: ["<<numlinks<<"]"<<endl;
			t1 = t2;							// 
			ioparm.timerFlag = true;				// 
		}									// 
		t2=time(&t2);							// 
		
	}
	fscan.close();
	cout << "  edgecount: ["<<numlinks<<"] total (first pass)"<<endl;
	gparm.maxid = numnodes+2;					// store maximum index
	elist = new edge [2*numlinks];				// create requisite number of edges
	int ecounter = 0;							// index of next edge of elist to be used

	// Now that we know numnodes, we can allocate the space for the sparse matrix, and
	// then reparse the file, adding edges as necessary.
	cout << " allocating space for network." << endl;
	e        = new  edge [gparm.maxid];			// (unordered) sparse adjacency matrix
	last     = new edge* [gparm.maxid];			// list of pointers to the last edge in each row
	numnodes = 0;								// numnodes now counts number of actual used node ids
	numlinks = 0;								// numlinks now counts number of bi-directional edges created
	ioparm.timerFlag = false;					// reset timer
	
	cout << " reparsing the input file to build network data structure." << endl;
	cout << "  edgecount: [0]"<<endl;
	ifstream fin(ioparm.f_input.c_str(), ios::in);
	while (fin >> s >> f) {
		s++; f++;								// increment s,f to prevent using e[0]
		if (f < s) { t = s; s = f; f = t; }		// guarantee s < f
		numlinks++;							// increment link count (preemptive)
		if (e[s].so == 0) {						// if first edge with s, add s and (s,f)
			e[s].so = s;						// 
			e[s].si = f;						// 
			last[s] = &e[s];					//    point last[s] at self
			numnodes++;						//    increment node count
		} else {								//    try to add (s,f) to s-edgelist
			current = &e[s];					// 
			existsFlag = false;					// 
			while (current != NULL) {			// check if (s,f) already in edgelist
				if (current->si==f) {			// 
					existsFlag = true;			//    link already exists
					numlinks--;				//    adjust link-count downward
					break;					// 
				}							// 
				current = current->next;			//    look at next edge
			}								// 
			if (!existsFlag) {					// if not already exists, append it
				newedge = &elist[ecounter++];		//    grab next-free-edge
				newedge -> so = s;				// 
				newedge -> si = f;				// 
				last[s] -> next = newedge;		//    append newedge to [s]'s list
				last[s]         = newedge;		//    point last[s] to newedge
			}								// 
		}									// 
		
		if (e[f].so == 0) {						// if first edge with f, add f and (f,s)
			e[f].so = f;						// 
			e[f].si = s;						// 
			last[f] = &e[f];					//    point last[s] at self
			numnodes++;						//    increment node count
		} else {								// try to add (f,s) to f-edgelist
			if (!existsFlag) {					//    if (s,f) wasn't in s-edgelist, then
				newedge = &elist[ecounter++];		//       (f,s) not in f-edgelist
				newedge -> so = f;				// 
				newedge -> si = s;				// 
				last[f] -> next = newedge;		//    append newedge to [f]'s list
				last[f]		 = newedge;		//    point last[f] to newedge
			}								// 
		}									
		existsFlag = false;						// reset existsFlag
		if (t2-t1>ioparm.timer) {				// check timer; if necessarsy, display
			cout << "  edgecount: ["<<numlinks<<"]"<<endl;
			t1 = t2;							// 
			ioparm.timerFlag = true;				// 
		}									// 
		t2=time(&t2);							// 
		
	}
	cout << "  edgecount: ["<<numlinks<<"] total (second pass)"<<endl;
	fin.close();
	
	// Now we record our work in the parameters file, and exit.
	ofstream fout(ioparm.f_parm.c_str(), ios::app);
	fout << "---NET_STATS----\n";
	fout << "MAXID-----:\t" << gparm.maxid-2 << "\n";
	fout << "NUMNODES--:\t" << numnodes << "\n";
	fout << "NUMEDGES--:\t" << numlinks << "\n";
	fout.close();

	gparm.m = numlinks;							// store actual number of edges created
	gparm.n = numnodes;							// store actual number of nodes used
	return;
}

//----------------------------------------------------------------------------//
// records the agglomerated list of indices for each valid community 
//----------------------------------------------------------------------------//

void recordGroupLists() {

	list *current;
	ofstream fgroup(ioparm.f_group.c_str(), ios::trunc);
	for (int i=0; i<gparm.maxid; i++) {
		if (c[i].valid) {
			fgroup << "GROUP[ "<<i-1<<" ][ "<<c[i].size<<" ]\n";   // external format
			current = c[i].members;
			while (current != NULL) {
				fgroup << current->index-1 << "\n";			// external format
				current = current->next;				
			}
		}
	}
	fgroup.close();
	
	return;
}

//----------------------------------------------------------------------------//
// records the network as currently agglomerated
//----------------------------------------------------------------------------//
void recordNetwork() {

	dpair *list, *current, *temp;
	
	ofstream fnet(ioparm.f_net.c_str(), ios::trunc);
	for (int i=0; i<gparm.maxid; i++) {
		if (dq[i].heap_ptr != NULL) {
			list    = dq[i].v->returnTreeAsList();			// get a list of items in dq[i].v
			current = list;							// store ptr to head of list
			while (current != NULL) {
				//		source		target		weight    (external representation)
				fnet << i-1 << "\t" << current->x-1 << "\t" << current->y << "\n";

				temp = current;						// clean up memory and move to next
				current = current->next;
				delete temp;				
			}
		}		
	}
	fnet.close();
	
	return;
}

}
//----------------------------------------------------------------------------//

} // namespace fastcommunity
