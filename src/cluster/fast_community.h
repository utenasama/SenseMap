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
#include <vector>
#include <string>
#include "stdlib.h"
#include "time.h"
#include "math.h"

#include "maxheap.h"
#include "vektor.h"

#include "community_tree.h"
#include "container/scene_graph_container.h"
#include "graph/correspondence_graph.h"
#include "graph/scene_clustering.h"

namespace fastcommunity {

void CommunityDetection(
	const sensemap::SceneClustering::Options& options,
	const std::shared_ptr<sensemap::SceneGraphContainer>& scene_graph_container,
	const std::set<uint32_t>& valid_image_ids,
	std::vector<std::vector<uint32_t> > &clusters,
	std::vector<std::unordered_set<uint32_t> > &overlaps);

void CommunityAddOverlap(
	const sensemap::SceneClustering::Options& options,
	const std::shared_ptr<sensemap::CorrespondenceGraph> &correspondence_graph,
	std::vector<std::vector<uint32_t> > &clusters,
	std::vector<std::unordered_set<uint32_t> > &overlaps);
} // namespace fastcommunity