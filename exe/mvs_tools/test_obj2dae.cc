//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <ctime>

#include "util/obj.h"
#include "util/misc.h"
#include "util/tinyxml.h"
#include "util/roi_box.h"
#include "base/common.h"
#include "util/exception_handler.h"

#include "base/version.h"
#include "../Configurator_yaml.h"


using namespace std;
using namespace sensemap;

#define EPSILON std::numeric_limits<float>::epsilon()
#define MAX_INT std::numeric_limits<int>::max()

std::string configuration_file_path;

class XMLFile
{
public:
	XMLFile(const char *xmlFileName);
	~XMLFile();

public:
	void TransObj2Dae(TriangleMesh obj);
	void ReadDae();
private:
	char *m_xmlFileName;
	TiXmlDocument *m_pDocument;
	TiXmlDeclaration *m_pDeclaration;
};

XMLFile::XMLFile(const char *xmlFileName)
{
	m_xmlFileName = new char [200];
	strcpy(m_xmlFileName, xmlFileName);
	m_pDocument = NULL;
	m_pDeclaration = NULL;
}

XMLFile::~XMLFile()
{
	if (m_xmlFileName != NULL)
		delete m_xmlFileName;

	if (m_pDocument != NULL)
		delete m_pDocument;

	// if (m_pDeclaration != NULL)
	// 	delete m_pDeclaration;
}

void XMLFile::TransObj2Dae(TriangleMesh obj)
{
	//创建XML文档指针
	m_pDocument = new TiXmlDocument(m_xmlFileName);
	if (NULL == m_pDocument)
	{
		return;
	}

	//声明XML
	m_pDeclaration = new TiXmlDeclaration("1.0", "UTF-8", "");
	if (NULL == m_pDeclaration)
	{
		return;
	}
	m_pDocument->LinkEndChild(m_pDeclaration);

	//创建根节点
	string xmlFileName = m_xmlFileName;
	string rootName = "COLLADA";//root noed named COLLADA 
	TiXmlElement *pRoot = new TiXmlElement(rootName.c_str());
	if (NULL == pRoot)
	{
		return;
	}
	//关联XML文档，成为XML文档的根节点
	m_pDocument->LinkEndChild(pRoot);
    pRoot->SetAttribute("xmlns","http://www.collada.org/2005/11/COLLADASchema");
    pRoot->SetAttribute("version", "1.4.1");

	//Creat asset Node
	TiXmlElement *pAsset = new TiXmlElement("asset");
	if (NULL == pAsset)
	{
		return;
	}
    pRoot->LinkEndChild(pAsset);

	TiXmlElement *pContributor  = new TiXmlElement("contributor");
	pAsset->LinkEndChild(pContributor);

    TiXmlElement *pAuthor  = new TiXmlElement("author");
	TiXmlText *pAuthorText = new TiXmlText("SenseTime Group");
	pAuthor->LinkEndChild(pAuthorText);
    pContributor->LinkEndChild(pAuthor);

    TiXmlElement *pAuthoringTool  = new TiXmlElement("authoring_tool");
	TiXmlText *pAuthoringToolText = new TiXmlText("SenseTime Group | SenseMap");
	pAuthoringTool->LinkEndChild(pAuthoringToolText);
    pContributor->LinkEndChild(pAuthoringTool);

	// system time
	time_t now = time(0);
    string dt = ctime(&now);
	string dt_sub = dt.substr(0, 24);
	// string dt_sub = dt.substr(0, dt.find("&#x0A"));

    TiXmlElement *pCreated  = new TiXmlElement("created");
	TiXmlText *pCreatedText = new TiXmlText(dt_sub.c_str());
	pCreated->LinkEndChild(pCreatedText);
    pAsset->LinkEndChild(pCreated);

    TiXmlElement *pModified  = new TiXmlElement("modified");
	TiXmlText *pModifiedText = new TiXmlText(dt_sub.c_str());
	pModified->LinkEndChild(pModifiedText);
    pAsset->LinkEndChild(pModified);


    TiXmlElement *pUpAxis  = new TiXmlElement("up_axis");
	TiXmlText *pUpAxisText = new TiXmlText("Y_UP");
	pUpAxis->LinkEndChild(pUpAxisText);
    pAsset->LinkEndChild(pUpAxis);

	// Creat library_geometries Nod
	TiXmlElement *pLibraryGeometries = new TiXmlElement("library_geometries");
	if (NULL == pLibraryGeometries)
	{
		return;
	}
	pRoot->LinkEndChild(pLibraryGeometries);

	int shapes = 0;
	int i_shape = 0;{
	// for (int i_shape = 0; i_shape < shapes; i_shape++){
		string shape_name = "shape" +  to_string(i_shape);
		string shape_id = shape_name + "-lib";

		TiXmlElement *pGeometry  = new TiXmlElement("geometry");
		pGeometry->SetAttribute("id", shape_id.c_str());
		pGeometry->SetAttribute("name", shape_name.c_str());
		pLibraryGeometries->LinkEndChild(pGeometry);

		TiXmlElement *pMesh  = new TiXmlElement("mesh");
		pGeometry->LinkEndChild(pMesh);

		// source position
		string Positions_id = shape_id + "-positions";
		TiXmlElement *pSourcePositions  = new TiXmlElement("source");
		pSourcePositions->SetAttribute("id", Positions_id.c_str());
		pSourcePositions->SetAttribute("name", "position");
		pMesh->LinkEndChild(pSourcePositions);

		int vtxs_size = obj.vertices_.size();	
		stringstream strPositionArray;
		for (int i = 0; i < vtxs_size; i++){
			strPositionArray << obj.vertices_.at(i)(0) << " "  
							 << obj.vertices_.at(i)(1) << " "
							 << obj.vertices_.at(i)(2) << " ";
		}
		string PositionArray_id = Positions_id + "-array";
		TiXmlElement *pSourcePositionArray  = new TiXmlElement("float_array");
		pSourcePositionArray->SetAttribute("id", PositionArray_id.c_str());
		pSourcePositionArray->SetAttribute("count", to_string(vtxs_size * 3).c_str());
		TiXmlText *pSourcePositionArrayText = new TiXmlText(strPositionArray.str().c_str());
		pSourcePositionArray->LinkEndChild(pSourcePositionArrayText);
		pSourcePositions->LinkEndChild(pSourcePositionArray);

		TiXmlElement *PositionTechniqueCommon = new TiXmlElement("technique_common");
		pSourcePositions->LinkEndChild(PositionTechniqueCommon);
		{
			TiXmlElement *PositionTCAccessor = new TiXmlElement("accessor");
			PositionTCAccessor->SetAttribute("count", to_string(vtxs_size).c_str());
			PositionTCAccessor->SetAttribute("source", ("#" + PositionArray_id).c_str());
			PositionTCAccessor->SetAttribute("stride", to_string(3).c_str());		
			PositionTechniqueCommon->LinkEndChild(PositionTCAccessor);
			{
				TiXmlElement *PositionAParamX = new TiXmlElement("param");
				PositionAParamX->SetAttribute("name", "X");
				PositionAParamX->SetAttribute("type", "float");
				PositionTCAccessor->LinkEndChild(PositionAParamX);
				TiXmlElement *PositionAParamY = new TiXmlElement("param");
				PositionAParamY->SetAttribute("name", "Y");
				PositionAParamY->SetAttribute("type", "float");
				PositionTCAccessor->LinkEndChild(PositionAParamY);
				TiXmlElement *PositionAParamZ = new TiXmlElement("param");
				PositionAParamZ->SetAttribute("name", "Z");
				PositionAParamZ->SetAttribute("type", "float");
				PositionTCAccessor->LinkEndChild(PositionAParamZ);
			}
		}

		// source normal
		string Normals_id = shape_id + "-normals";
		TiXmlElement *pSourceNormals  = new TiXmlElement("source");
		pSourceNormals->SetAttribute("id", Normals_id.c_str());
		pSourceNormals->SetAttribute("name", "normal");
		pMesh->LinkEndChild(pSourceNormals);

		int vtxsNormal_size = obj.face_normals_.size();	
		stringstream strNormalArray;
		for (int i = 0; i < vtxsNormal_size; i++){
			strNormalArray << obj.face_normals_.at(i)(0) << " "  
							 << obj.face_normals_.at(i)(1) << " "
							 << obj.face_normals_.at(i)(2) << " ";
		}

		string NormalArray_id = Normals_id + "-array";
		TiXmlElement *pSourceNormalArray  = new TiXmlElement("float_array");
		pSourceNormalArray->SetAttribute("id", NormalArray_id.c_str());
		pSourceNormalArray->SetAttribute("count", to_string(vtxsNormal_size * 3).c_str());
		TiXmlText *pSourceNormalArrayText = new TiXmlText(strNormalArray.str().c_str());
		pSourceNormalArray->LinkEndChild(pSourceNormalArrayText);
		pSourceNormals->LinkEndChild(pSourceNormalArray);

		TiXmlElement *NormalTechniqueCommon = new TiXmlElement("technique_common");
		pSourceNormals->LinkEndChild(NormalTechniqueCommon);
		{
			TiXmlElement *NormalTCAccessor = new TiXmlElement("accessor");
			NormalTCAccessor->SetAttribute("count", to_string(vtxsNormal_size).c_str());
			NormalTCAccessor->SetAttribute("source", ("#" + NormalArray_id).c_str());
			NormalTCAccessor->SetAttribute("stride", to_string(3).c_str());		
			NormalTechniqueCommon->LinkEndChild(NormalTCAccessor);
			{
				TiXmlElement *NormalAParamX = new TiXmlElement("param");
				NormalAParamX->SetAttribute("name", "X");
				NormalAParamX->SetAttribute("type", "float");
				NormalTCAccessor->LinkEndChild(NormalAParamX);
				TiXmlElement *NormalAParamY = new TiXmlElement("param");
				NormalAParamY->SetAttribute("name", "Y");
				NormalAParamY->SetAttribute("type", "float");
				NormalTCAccessor->LinkEndChild(NormalAParamY);
				TiXmlElement *NormalAParamZ = new TiXmlElement("param");
				NormalAParamZ->SetAttribute("name", "Z");
				NormalAParamZ->SetAttribute("type", "float");
				NormalTCAccessor->LinkEndChild(NormalAParamZ);
			}
		}

		// source vcolor
		string Vcolor_id = shape_id + "-vcolor";
		TiXmlElement *pSourceVcolor  = new TiXmlElement("source");
		pSourceVcolor->SetAttribute("id", Vcolor_id.c_str());
		pSourceVcolor->SetAttribute("name", "vcolor");
		pMesh->LinkEndChild(pSourceVcolor);

		int vtxsVcolor_size = obj.vertex_colors_.size();	
		stringstream strVcolorArray;
		for (int i = 0; i < vtxsVcolor_size; i++){
			Eigen::Vector3d color = obj.vertex_colors_.at(i).normalized();
			strVcolorArray << obj.vertex_colors_.at(i)(0)/255.0 << " "  
						   << obj.vertex_colors_.at(i)(1)/255.0 << " "
						   << obj.vertex_colors_.at(i)(2)/255.0 << " "
						   << 1/255.0 << " ";
		}

		string VcolorArray_id = Vcolor_id + "-array";
		TiXmlElement *pSourceVcolorArray  = new TiXmlElement("float_array");
		pSourceVcolorArray->SetAttribute("id", VcolorArray_id.c_str());
		pSourceVcolorArray->SetAttribute("count", to_string(vtxsVcolor_size * 4).c_str());
		TiXmlText *pSourceVcolorArrayText = new TiXmlText(strVcolorArray.str().c_str());
		pSourceVcolorArray->LinkEndChild(pSourceVcolorArrayText);
		pSourceVcolor->LinkEndChild(pSourceVcolorArray);

		TiXmlElement *VcolorTechniqueCommon = new TiXmlElement("technique_common");
		pSourceVcolor->LinkEndChild(VcolorTechniqueCommon);
		{
			TiXmlElement *VcolorTCAccessor = new TiXmlElement("accessor");
			VcolorTCAccessor->SetAttribute("count", to_string(vtxs_size).c_str());
			VcolorTCAccessor->SetAttribute("source", ("#" + VcolorArray_id).c_str());
			VcolorTCAccessor->SetAttribute("stride", to_string(4).c_str());		
			VcolorTechniqueCommon->LinkEndChild(VcolorTCAccessor);
			{
				TiXmlElement *VcolorAParamR = new TiXmlElement("param");
				VcolorAParamR->SetAttribute("name", "R");
				VcolorAParamR->SetAttribute("type", "float");
				VcolorTCAccessor->LinkEndChild(VcolorAParamR);
				TiXmlElement *VcolorAParamG = new TiXmlElement("param");
				VcolorAParamG->SetAttribute("name", "G");
				VcolorAParamG->SetAttribute("type", "float");
				VcolorTCAccessor->LinkEndChild(VcolorAParamG);
				TiXmlElement *VcolorAParamB = new TiXmlElement("param");
				VcolorAParamB->SetAttribute("name", "B");
				VcolorAParamB->SetAttribute("type", "float");
				VcolorTCAccessor->LinkEndChild(VcolorAParamB);
				TiXmlElement *VcolorAParamA = new TiXmlElement("param");
				VcolorAParamA->SetAttribute("name", "A");
				VcolorAParamA->SetAttribute("type", "float");
				VcolorTCAccessor->LinkEndChild(VcolorAParamA);
			}
		}
		
		TiXmlElement *pMeshVertices  = new TiXmlElement("vertices");
		pMeshVertices->SetAttribute("id", (shape_id + "-vertices").c_str());
		pMesh->LinkEndChild(pMeshVertices);
		TiXmlElement *pMeshVerticesInput  = new TiXmlElement("input");
		pMeshVerticesInput->SetAttribute("semantic", "POSITION");
		pMeshVerticesInput->SetAttribute("source", ("#" + Positions_id).c_str());
		pMeshVertices->LinkEndChild(pMeshVerticesInput);

		int vface_size = obj.faces_.size();	
		stringstream strFaceArray;
		for (int i = 0; i < vface_size; i++){
			strFaceArray << obj.faces_.at(i)(0) << " "  
						   << obj.faces_.at(i)(0) << " "
						   << i << " "  
						   << obj.faces_.at(i)(1) << " "
						   << obj.faces_.at(i)(1) << " "
						   << i << " " 
						   << obj.faces_.at(i)(2) << " "
						   << obj.faces_.at(i)(2) << " "
						   << i << " ";
		}
		TiXmlElement *pMeshTriangles  = new TiXmlElement("triangles");
		pMeshTriangles->SetAttribute("count", to_string(vface_size).c_str());
		pMesh->LinkEndChild(pMeshTriangles);
		TiXmlElement *pMeshTrianglesInput1 = new TiXmlElement("input");
		pMeshTrianglesInput1->SetAttribute("offset", "0");
		pMeshTrianglesInput1->SetAttribute("semantic", "VERTEX");
		pMeshTrianglesInput1->SetAttribute("source", "#shape0-lib-vertices");
		pMeshTriangles->LinkEndChild(pMeshTrianglesInput1);
		TiXmlElement *pMeshTrianglesInput2 = new TiXmlElement("input");
		pMeshTrianglesInput2->SetAttribute("offset", "1");
		pMeshTrianglesInput2->SetAttribute("semantic", "COLOR");
		pMeshTrianglesInput2->SetAttribute("source", "#shape0-lib-vcolor");
		pMeshTriangles->LinkEndChild(pMeshTrianglesInput2);
		TiXmlElement *pMeshTrianglesInput3 = new TiXmlElement("input");
		pMeshTrianglesInput3->SetAttribute("offset", "2");
		pMeshTrianglesInput3->SetAttribute("semantic", "NORMAL");
		pMeshTrianglesInput3->SetAttribute("source", "#shape0-lib-normals");
		pMeshTriangles->LinkEndChild(pMeshTrianglesInput3);
		TiXmlElement *pMeshTrianglesInputEnd = new TiXmlElement("p");
		TiXmlText *pMeshTrianglesFaceArrayText = new TiXmlText(strFaceArray.str().c_str());
		pMeshTrianglesInputEnd->LinkEndChild(pMeshTrianglesFaceArrayText);
		pMeshTriangles->LinkEndChild(pMeshTrianglesInputEnd);

	}

	// Creat library_visual_scenes Node
	TiXmlElement *pLibraryVisualScenes = new TiXmlElement("library_visual_scenes");
	if (NULL == pLibraryVisualScenes)
	{
		return;
	}
	pRoot->LinkEndChild(pLibraryVisualScenes);

	TiXmlElement *pVisualScene  = new TiXmlElement("visual_scene");
	pVisualScene->SetAttribute("id", "VisualSceneNode");
	pVisualScene->SetAttribute("name", "VisualScene");
	pLibraryVisualScenes->LinkEndChild(pVisualScene);

	TiXmlElement *pVisualSceneNode  = new TiXmlElement("node");
	pVisualSceneNode->SetAttribute("id", "node");
	pVisualSceneNode->SetAttribute("name", "node");
	pVisualScene->LinkEndChild(pVisualSceneNode);
	TiXmlElement *pVSNodeInstanceGeometry  = new TiXmlElement("instance_geometry");
	pVSNodeInstanceGeometry->SetAttribute("url", "#shape0-lib");
	pVisualSceneNode->LinkEndChild(pVSNodeInstanceGeometry);

	// Creat scenes Node
	TiXmlElement *pScene = new TiXmlElement("scene");
	if (NULL == pScene)
	{
		return;
	}
	pRoot->LinkEndChild(pScene);

	TiXmlElement *pScenesInstanceVisualScene  = new TiXmlElement("instance_visual_scene");
	pScenesInstanceVisualScene->SetAttribute("url", "#VisualSceneNode");
	pScene->LinkEndChild(pScenesInstanceVisualScene);

	m_pDocument->SaveFile(m_xmlFileName);
}

//读取XML文件完整内容
void XMLFile::ReadDae()
{
	if (m_xmlFileName == NULL)
	{
		cout << " null " << endl;
		return;
	}
	m_pDocument->LoadFile(m_xmlFileName);
	m_pDocument->Print();
}

struct MeshClusterOptions{
	int min_faces_per_cluster = 50000;   // Minimum number of faces for each mesh cluster.
    int max_faces_per_cluster = 100000;   // Maximum number of faces for each mesh cluster.
    // cell_size_factor * [average spacing] is the size of cell in each dimension, usually for cases without scale.
    // If cell_size_factor <= 0, cell_size is used; else, compute cell_size using cell_size_factor.
    float cell_size_factor = 100.0f;
    // Size of cell in each dimension, usually for cases with scale and dimension.
    float cell_size = 6.0f;

    double valid_spacing_factor = 2.5;
};

bool WhetherMerge(const std::vector<Box> &cell_bound_box,
				  const std::vector<Box> &cluster_bound_box,
                  const int cell_idx, const int merge_cluster_idx){
    bool x_flag = 
        (std::abs(cell_bound_box[cell_idx].x_min - 
        cluster_bound_box[merge_cluster_idx].x_min) < EPSILON) && 
        (std::abs(cell_bound_box[cell_idx].x_max - 
        cluster_bound_box[merge_cluster_idx].x_max) < EPSILON) &&
        ((std::abs(cell_bound_box[cell_idx].y_min - 
        cluster_bound_box[merge_cluster_idx].y_max) < EPSILON) ||
        std::abs(cell_bound_box[cell_idx].y_max - 
        cluster_bound_box[merge_cluster_idx].y_min) < EPSILON);
    bool y_flag = 
        (std::abs(cell_bound_box[cell_idx].y_min - 
        cluster_bound_box[merge_cluster_idx].y_min) < EPSILON) && 
        (std::abs(cell_bound_box[cell_idx].y_max - 
        cluster_bound_box[merge_cluster_idx].y_max) < EPSILON) &&
        ((std::abs(cell_bound_box[cell_idx].x_min - 
        cluster_bound_box[merge_cluster_idx].x_max) < EPSILON) ||
        std::abs(cell_bound_box[cell_idx].x_max - 
        cluster_bound_box[merge_cluster_idx].x_min) < EPSILON);

    if (x_flag || y_flag){
        // std::cout << "cell idx:" << cell_idx << "\tcluster_idx:"
        //           << merge_cluster_idx << std::endl;
        return true;
    }
    return false;
}

std::size_t GridCluster(const struct MeshClusterOptions& options,
						std::vector<int> &cell_cluster_map,
						std::vector<std::size_t> &cell_point_count,
						std::vector<std::size_t> &point_cell_map,
						std::vector<Eigen::Vector3f> &points,
						std::vector<Box> &cell_bound_box,
						const std::size_t grid_size_x,
						const std::size_t grid_size_y){
	std::size_t grid_slide = cell_bound_box.size();
    const int point_num = points.size();
	int max_points_num = 0;
	for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){
        while(cell_point_count[cell_idx] > options.max_faces_per_cluster){
            // if ((cell_bound_box[cell_idx].x_max - cell_bound_box[cell_idx].x_min) 
            //     < options_.max_cell_size * 0.5 &&
            //     (cell_bound_box[cell_idx].y_max - cell_bound_box[cell_idx].y_min) 
            //     < options_.max_cell_size * 0.5){
            //     break;
            // }

            cell_point_count.push_back(0);
            cell_bound_box.push_back(Box());
            if (((cell_bound_box[cell_idx].x_max - cell_bound_box[cell_idx].x_min) - 
                (cell_bound_box[cell_idx].y_max - cell_bound_box[cell_idx].y_min)) < 0){
                // Split in Y direction
                float split_y = (cell_bound_box[cell_idx].y_min + 
                                cell_bound_box[cell_idx].y_max) / 2;
                Box box1 = cell_bound_box[cell_idx];
                Box box2 = cell_bound_box[cell_idx];
                box1.y_max = split_y;
                box2.y_min = split_y;
                cell_bound_box[cell_idx] = box1;
                cell_bound_box[grid_slide] = box2;
                
                for(std::size_t i = 0; i < point_num; ++i){
                    if (point_cell_map[i] != cell_idx){
                        continue;
                    }
                    auto &point = points[i];
                    if (point.y() > split_y){
                        point_cell_map[i] = grid_slide;
                        cell_point_count[cell_idx]--;
                        cell_point_count[grid_slide]++;
                    }
                }
            } else {
                // Split in X direction
                float split_x = (cell_bound_box[cell_idx].x_min + 
                                cell_bound_box[cell_idx].x_max) / 2;
                Box box1 = cell_bound_box[cell_idx];
                Box box2 = cell_bound_box[cell_idx];
                box1.x_max = split_x;
                box2.x_min = split_x;
                cell_bound_box[cell_idx] = box1;
                cell_bound_box[grid_slide] = box2;
                
                for(std::size_t i = 0; i < point_num; ++i){
                    if (point_cell_map[i] != cell_idx){
                        continue;
                    }
                    auto &point = points[i];
                    if (point.x() > split_x){
                        point_cell_map[i] = grid_slide;
                        cell_point_count[cell_idx]--;
                        cell_point_count[grid_slide]++;
                    }
                }
            }
            grid_slide++;
        }
		max_points_num = 
			max_points_num > cell_point_count[cell_idx] ? max_points_num : cell_point_count[cell_idx];
        
		std::cout << "Cell "<< cell_idx << ": " 
		<< "\tpoint size: " << cell_point_count[cell_idx] << std::endl;
    }

    std::size_t cell_num = cell_point_count.size();
    cell_cluster_map.resize(cell_num);
    std::fill(cell_cluster_map.begin(), cell_cluster_map.end(), -1);

    std::vector<std::pair<std::size_t, std::size_t> > dense_cells;
    dense_cells.reserve(cell_num);
    std::vector<std::size_t> sparse_cells;
    sparse_cells.reserve(cell_num);
    std::vector<unsigned char> cell_type_map(cell_num, 0);
    for (std::size_t i = 0; i < cell_num; i++) {
        if (cell_point_count[i] < options.min_faces_per_cluster &&
		    cell_point_count[i] != max_points_num) {
            sparse_cells.push_back(i);
            cell_type_map[i] = 128;
            continue;
        }

        dense_cells.emplace_back(cell_point_count[i], i);
        cell_type_map[i] = 255;
    }
    dense_cells.shrink_to_fit();
    sparse_cells.shrink_to_fit();

    std::size_t cluster_idx = 0;
    std::size_t dense_cell_num = dense_cells.size();
	std::vector<Box> cluster_bound_box;
    cluster_bound_box.resize(dense_cell_num);
    std::vector<std::size_t> cluster_point_count;
    std::vector<std::vector<std::size_t> > cluster_cells_map;



    for (std::size_t i = 0; i < dense_cell_num; i++) {
        int cell_idx = dense_cells[i].second;
        if (cell_cluster_map[cell_idx] != -1) {
            continue;
        }

        std::size_t num_visited_points = 0;
        cluster_cells_map.push_back(std::vector<std::size_t>());
        auto &cluster_cells = cluster_cells_map.back();

        cell_cluster_map[cell_idx] = cluster_idx;
        cluster_cells.push_back(cell_idx);
        num_visited_points = cell_point_count[cell_idx];

        cluster_bound_box[i] = cell_bound_box[cell_idx];

        cluster_point_count.push_back(num_visited_points);
        // std::cout << num_visited_points << " points clustered" << std::endl;
        cluster_idx++;
    }

    std::size_t cluster_num = cluster_idx;
    std::vector<std::size_t> cluster_idx_map;
    cluster_idx_map.reserve(cluster_num);
    for (std::size_t i = 0; i < cluster_num; i++) {
        cluster_idx_map.push_back(i);
    }
    cluster_idx_map.shrink_to_fit();

    for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
        const auto &cluster_idx = cluster_idx_map[i];
        cluster_point_count[i] = cluster_point_count[cluster_idx];
        cluster_cells_map[i] = cluster_cells_map[cluster_idx];
        for (auto cell_idx : cluster_cells_map[i]) {
            cell_cluster_map[cell_idx] = i;
        }
    }
    
    for (auto cell_idx : sparse_cells) {
        bool merge_flag = false;
        int merge_cluster_idx = -1;
        for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
            merge_cluster_idx = i;
            if (!WhetherMerge(cell_bound_box, cluster_bound_box, 
							  cell_idx, merge_cluster_idx)) {
                continue;
            }
			if (cell_point_count[cell_idx] + cluster_point_count[merge_cluster_idx] 
				> options.max_faces_per_cluster){
				continue;
			}

            cell_cluster_map[cell_idx] = merge_cluster_idx;
            cluster_point_count[merge_cluster_idx] += cell_point_count[cell_idx];
            cluster_cells_map[merge_cluster_idx].push_back(cell_idx);
            cluster_bound_box[merge_cluster_idx] += cell_bound_box[cell_idx];
            cell_type_map[cell_idx] = 255;
            merge_flag = true;
            std::cout << "merge: cell_idx: " << cell_idx << "to Cluster " << merge_cluster_idx << std::endl;
            break;
        }
          
        if (!merge_flag){            
            std::size_t num_visited_points = 0;
            cluster_cells_map.push_back(std::vector<std::size_t>());
            auto &cluster_cells = cluster_cells_map.back();

            cell_cluster_map[cell_idx] = cluster_idx;
            cluster_cells.push_back(cell_idx);
            num_visited_points = cell_point_count[cell_idx];

            cluster_idx_map.push_back(cluster_idx);

            cluster_point_count.push_back(num_visited_points);
            cluster_bound_box.push_back(cell_bound_box[cell_idx]);
            cell_type_map[cell_idx] = 255;
            // std::cout << num_visited_points << " points clustered" << std::endl;  

            cluster_idx++;
            cluster_num = cluster_idx;
        }
    }

    cluster_num = cluster_idx_map.size();
    cluster_point_count.resize(cluster_num);
    cluster_cells_map.resize(cluster_num);
    cluster_bound_box.resize(cluster_num);

	return cluster_num;
};

std::size_t Cluster(const struct MeshClusterOptions& options,
					std::vector<int> &face_cluster_map,
					const TriangleMesh& mesh){
	std::cout << "MeshCluster::Cluster" << std::endl;
	if (mesh.vertices_.empty()|| mesh.faces_.empty()) {
        std::vector<int>().swap(face_cluster_map);
        return 0;
    }

    // Compute Pivot.
    Eigen::Matrix3f pivot;
    Eigen::Vector3d centroid(Eigen::Vector3d::Zero());
    for (const auto &vertex : mesh.vertices_) {
        centroid += vertex;
    }
    std::size_t vertex_num = mesh.vertices_.size();
    centroid /= vertex_num;

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; ++i) {
        for (const auto &vertex : mesh.vertices_) {
            M(i, 0) += (vertex[i] - centroid[i]) * (vertex[0] - centroid[i]);
            M(i, 1) += (vertex[i] - centroid[i]) * (vertex[1] - centroid[i]);
            M(i, 2) += (vertex[i] - centroid[i]) * (vertex[2] - centroid[i]);
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    pivot = svd.matrixU().transpose();

    std::size_t face_num = mesh.faces_.size();
    std::vector<Eigen::Vector3f> transformed_tri_centroids(face_num);
    Eigen::Vector3f box_min, box_max;
    {
        auto &transformed_tri_centroid = transformed_tri_centroids[0];
        const auto &face = mesh.faces_[0];
        transformed_tri_centroid = pivot * (mesh.vertices_[face[0]] + mesh.vertices_[face[1]] + mesh.vertices_[face[2]]).cast<float>() / 3.0f;
        box_min = box_max = pivot * transformed_tri_centroid;
    }
    for (int i = 1; i < face_num; ++i) {
        auto &transformed_tri_centroid = transformed_tri_centroids[i];
        const auto &face = mesh.faces_[i];
        transformed_tri_centroid = pivot * (mesh.vertices_[face[0]] + mesh.vertices_[face[1]] + mesh.vertices_[face[2]]).cast<float>() / 3.0f;
        
        box_min[0] = std::min(box_min[0], transformed_tri_centroid[0]);
        box_min[1] = std::min(box_min[1], transformed_tri_centroid[1]);
        box_min[2] = std::min(box_min[2], transformed_tri_centroid[2]);
        box_max[0] = std::max(box_max[0], transformed_tri_centroid[0]);
        box_max[1] = std::max(box_max[1], transformed_tri_centroid[1]);
        box_max[2] = std::max(box_max[2], transformed_tri_centroid[2]);
    }
	const float grid_size = face_num / options.max_faces_per_cluster;
    const float cell_size = std::sqrt((box_max.x() - box_min.x()) * 
                (box_max.y() - box_min.y()) / grid_size);
    const std::size_t grid_size_x = static_cast<std::size_t>((box_max.x() - box_min.x()) / cell_size) + 1;
    const std::size_t grid_size_y = static_cast<std::size_t>((box_max.y() - box_min.y()) / cell_size) + 1;
    const std::size_t grid_side = grid_size_x;
    const std::size_t grid_slide = grid_side * grid_size_y;


    double delt_x = (grid_size_x * cell_size - box_max.x() + box_min.x()) / 2;
    double delt_y = (grid_size_y * cell_size - box_max.y() + box_min.y()) / 2;
    box_min[0] -= delt_x;
    box_min[1] -= delt_y;
    box_max[0] += delt_x;
    box_max[1] += delt_y;

    std::vector<std::size_t> cell_face_count(grid_slide, 0);
    std::vector<std::size_t> face_cell_map(face_num);

    for (std::size_t i = 0; i < face_num; ++i) {
        const auto &transformed_tri_centroid = transformed_tri_centroids[i];
        std::size_t x_cell = static_cast<std::size_t>((transformed_tri_centroid.x() - box_min.x()) / cell_size);
        std::size_t y_cell = static_cast<std::size_t>((transformed_tri_centroid.y() - box_min.y()) / cell_size);

        std::size_t cell_idx = y_cell * grid_side + x_cell;
        cell_face_count[cell_idx]++;
        face_cell_map[i] = cell_idx;
    }


    // bound box
    std::vector<Box> cell_bound_boxs(grid_slide);
    for (std::size_t cell_idx = 0; cell_idx < grid_slide; cell_idx++){
        int y_cell = cell_idx / grid_size_x;
        int x_cell = cell_idx % grid_size_x;

        Box box;
        box.x_min = x_cell * cell_size + box_min.x();
        box.y_min = y_cell * cell_size + box_min.y();
        box.x_max = (x_cell + 1) * cell_size + box_min.x();
        box.y_max = (y_cell + 1) * cell_size + box_min.y();
        // std::cout << "id: " << cell_idx << "(" << box.x_min << " " << box.y_min << ")-(" << box.x_max << " " << box.y_max << ")" << std::endl;
        box.rot;
        cell_bound_boxs[cell_idx] = box;
    }

    std::vector<int> cell_cluster_map;
	std::size_t cluster_num 
		= GridCluster(options, cell_cluster_map, cell_face_count, face_cell_map, 
			transformed_tri_centroids, cell_bound_boxs, grid_size_x, grid_size_y);

    face_cluster_map.resize(face_num);
    memset(face_cluster_map.data(), -1, face_num * sizeof(int));
    for (std::size_t i = 0; i < face_num; ++i) {
        face_cluster_map[i] = cell_cluster_map[face_cell_map[i]];
    }

	return cluster_num;
};

void MeshCluster(const struct MeshClusterOptions& options,
				 const TriangleMesh& mesh,
				 std::vector<TriangleMesh>& v_cluster_mesh){
	std::size_t vertex_num = mesh.vertices_.size();
    std::size_t face_num = mesh.faces_.size();
    // std::cout << face_num << " faces to be clustered" << std::endl;

    std::vector<int> face_cluster_map;
    int cluster_num = Cluster(options, face_cluster_map, mesh);

    std::vector<std::vector<std::size_t> > clustered_faces(cluster_num);
    for (std::size_t i = 0; i < face_num; ++i) {
        auto cluster_idx = face_cluster_map[i];
        if (cluster_idx < 0) {
            continue;
        }

        clustered_faces[cluster_idx].emplace_back(i);
    }

	// Save cluster dae

    // Save clustered meshes.
	v_cluster_mesh.clear();
    for (std::size_t cluster_idx = 0; cluster_idx < cluster_num; cluster_idx++) {
		if (clustered_faces[cluster_idx].size() < 1){
			continue;
		}
        std::cout << "Cluster: " << cluster_idx 
			<< "\t face num: " << clustered_faces[cluster_idx].size() << std::endl;

        
        TriangleMesh cluster_mesh;
        cluster_mesh.vertices_.reserve(mesh.vertices_.size());
        cluster_mesh.vertex_normals_.reserve(mesh.vertex_normals_.size());
        cluster_mesh.vertex_colors_.reserve(mesh.vertex_colors_.size());
        cluster_mesh.vertex_labels_.reserve(mesh.vertex_labels_.size());
        cluster_mesh.faces_.reserve(mesh.faces_.size());
        cluster_mesh.face_normals_.reserve(mesh.face_normals_.size());
        for (auto face_id : clustered_faces[cluster_idx]) {
            cluster_mesh.faces_.push_back(mesh.faces_[face_id]);
        }
        if (!mesh.face_normals_.empty()) {
            for (auto face_id : clustered_faces[cluster_idx]) {
                cluster_mesh.face_normals_.push_back(mesh.face_normals_[face_id]);
            }
        }
        std::vector<int> vertex_idx_map(vertex_num, -1);
        for (auto face_id : clustered_faces[cluster_idx]) {
            const auto &face = mesh.faces_[face_id];
            vertex_idx_map[face[0]] = 0;
            vertex_idx_map[face[1]] = 0;
            vertex_idx_map[face[2]] = 0;
        }
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }

            vertex_idx_map[i] = cluster_mesh.vertices_.size();
            cluster_mesh.vertices_.push_back(mesh.vertices_[i]);
        }
        if (!mesh.vertex_normals_.empty()) {
            for (std::size_t i = 0; i < vertex_num; i++) {
                if (vertex_idx_map[i] == -1) {
                    continue;
                }

                cluster_mesh.vertex_normals_.push_back(mesh.vertex_normals_[i]);
            }
        }
        if (!mesh.vertex_colors_.empty()) {
            for (std::size_t i = 0; i < vertex_num; i++) {
                if (vertex_idx_map[i] == -1) {
                    continue;
                }

                cluster_mesh.vertex_colors_.push_back(mesh.vertex_colors_[i]);
            }
        }
        if (!mesh.vertex_labels_.empty()) {
            for (std::size_t i = 0; i < vertex_num; i++) {
                if (vertex_idx_map[i] == -1) {
                    continue;
                }

                cluster_mesh.vertex_labels_.push_back(mesh.vertex_labels_[i]);
            }
        }
        cluster_mesh.vertices_.shrink_to_fit();
        cluster_mesh.vertex_normals_.shrink_to_fit();
        cluster_mesh.vertex_colors_.shrink_to_fit();
        cluster_mesh.vertex_labels_.shrink_to_fit();
        cluster_mesh.faces_.shrink_to_fit();
        cluster_mesh.face_normals_.shrink_to_fit();
        for (auto &face : cluster_mesh.faces_) {
            face[0] = vertex_idx_map[face[0]];
            face[1] = vertex_idx_map[face[1]];
            face[2] = vertex_idx_map[face[2]];
        }

		v_cluster_mesh.push_back(cluster_mesh);
    }
};

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
	
	int max_faces_per_cluster = -1;
	if (argc == 4){
		max_faces_per_cluster = std::stoi(argv[3]);
	} else if (argc != 3){
		std::cout << "Error! Input: in-model-path out-model-path (max-faces-num) \n"
				  << "eg: /test_obj2dae ./workspace/0/dense/model.obj ./workspace/0/dense/model.dae 10000000" << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
	}

	const std::string in_model_path = std::string(argv[1]);
	std::string out_model_path = std::string(argv[2]);
	if (out_model_path.substr(out_model_path.length() - 4) == ".dae"){
		out_model_path = out_model_path.substr(0, out_model_path.length() - 4);
	}

	std::cout << "in_model_path: " << in_model_path << std::endl;

	if (out_model_path.empty()) {
		std::cerr << "Error! Output model path is empty!" << std::endl;
		return StateCode::INVALID_INPUT_PARAM;
	}

    // read obj file
    TriangleMesh obj_model;
    if (!ReadTriangleMeshObj(in_model_path, obj_model, true)) {
		return StateCode::INVALID_INPUT_PARAM;
	}
    // ReadTriangleMeshObj(in_model_path, obj_model, false);
	obj_model.ComputeNormals();

	if (max_faces_per_cluster > 0){
		MeshClusterOptions options;
		options.max_faces_per_cluster = max_faces_per_cluster;
		options.min_faces_per_cluster = max_faces_per_cluster * 2 / 3;
		std::vector<TriangleMesh> v_cluster_mesh;
			MeshCluster(options, obj_model, v_cluster_mesh);

		for (int i = 0; i < v_cluster_mesh.size(); i++){
			std::string out_cluster_model_path = out_model_path+"-"+to_string(i)+".dae";
			XMLFile xmlFile(out_cluster_model_path.c_str());
			xmlFile.TransObj2Dae(v_cluster_mesh.at(i));
			std::cout << "out_dae_model_path-" << i << ": " << out_cluster_model_path << std::endl;
		}
	} else {
		// transform and save dae file
		out_model_path = out_model_path+".dae";
		XMLFile xmlFile(out_model_path.c_str());
		xmlFile.TransObj2Dae(obj_model);
		std::cout << "out_dae_model_path: " << out_model_path << std::endl;
	}

	// // xmlFile.ReadDae();

	return 0;
}