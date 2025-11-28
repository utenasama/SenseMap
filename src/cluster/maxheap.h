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

#ifndef FAST_COMMUNITY_MAX_HEAP_H_
#define FAST_COMMUNITY_MAX_HEAP_H_

#include <iostream>

namespace fastcommunity {

// #if !defined(TUPLE_INCLUDED)
// #define TUPLE_INCLUDED
struct tuple {
	double    m;					// stored value
	int		i;					// row index
	int		j;					// column index
	int		k;					// heap index
};
// #endif

/*   Because using this heap requires us to be able to modify an arbitrary element's
	data in constant O(1) time, I use to tricky tactic of having elements in an array-
	based heap only contain addresses to the data, rather than the data itself. In this
	manner, an external program may freely modify an element of the heap, given that it
	possesses a pointer to said element (in an array-based heap, the addresses and the 
	value in that address are not bound and thus may change during the heapify() operation).
*/

struct hnode { tuple     *d; };
const int heapmin = 3;
//const double tiny = -4294967296.0;

class maxheap {
private:
	hnode    *A;					// maxheap array
	int       heaplimit;			// first unused element of heap array
	int		arraysize;			// size of array
	bool		isempty;				// T if heap is empty; F otherwise

	int		downsift(int i);		// sift A[i] down in heap
	int		upsift  (int i);		// sift A[i] up in heap
	int		left    (int i);		// returns index of left child
	int		right   (int i);		// returns index of right child
	int		parent  (int i);		// returns index of parent
	void		grow();				// increase size of array A
	void		shrink();				// decrease size of array A
  
public:
	maxheap();					// default constructor
	maxheap(int size);				// default constructor
	~maxheap();					// default destructor
	
	int		heapSize();							// returns heaplimit value
	bool		heapIsEmpty();							// returns isempty value
	tuple	*insertItem(const tuple newData);			// heap-inserts newData, returns the address of it
	tuple	popMaximum();							// removes and returns heap max, reheapifies
	tuple	returnMaximum();						// returns the heap max; no change to heap
	void		printHeap();							// displays contents of the heap
	void		printHeapTop10();						// displays top 10 entries in the heap
	void		updateItem(tuple *address, tuple newData);   // updates the value of the tuple at address
	void		updateItem(tuple *address, double newStored);// update only the stored value of tuple at address
	void		deleteItem(tuple *address);				// remove an item from the heap
	int		returnArraysize();						// 
	int		returnHeaplimit();						// 
	
};

} // namespace fastcommunity

#endif