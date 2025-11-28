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

#ifndef FAST_COMMUNITY_VEKTOR_H_
#define FAST_COMMUNITY_VEKTOR_H_

#include <iostream>

#include "maxheap.h"

namespace fastcommunity {

class dpair {
public:
	int x; double y; dpair *next;
	dpair() { x = 0; y = 0.0; next = NULL; }
	~dpair() {}
};
struct dppair { dpair *head; dpair *tail; };

class element {
public:
	int		key;					// binary-tree key
	double    stored;				// additional stored value (associated with key)
	tuple	*heap_ptr;			// pointer to element's location in vektor max-heap
	
	bool		color;				// F: BLACK
								// T: RED
	element   *parent;				// pointer to parent node
	element   *left;				// pointer for left subtree
	element   *right;				// pointer for right subtree
	
	element() {    key = 0; stored = -4294967296.0; color = false;
					parent  = NULL; left  = NULL; right  = NULL; }
	~element() {}
};

/*   This vector implementation is a pair of linked data structures: a red-black balanced binary
	tree data structure and a maximum heap. This pair allows us to find a stored element in time
	O(log n), find the maximum element in time O(1), update the maximum element in time O(log n),
	delete an element in time O(log n), and insert an element in time O(log n). These complexities
	allow a much faster implementation of the fastCommunity algorithm. If we dispense with the
	max-heap, then some operations related to updating the maximum stored value can take up to O(n),
	which is potentially very slow.

	Both the red-black balanced binary tree and the max-heap implementations are custom-jobs. Note
	that the key=0 is assumed to be a special value, and thus you cannot insert such an item. 
	Beware of this limitation.
*/

class vektor {
private:
	element    *root;				// binary tree root
	element    *leaf;				// all leaf nodes
	maxheap    *heap;				// max-heap of elements in vektor
	int		 support;				// number of nodes in the tree

	void		rotateLeft(element *x);						// left-rotation operator
	void		rotateRight(element *y);						// right-rotation operator
	void		insertCleanup(element *z);					// house-keeping after insertion
	void		deleteCleanup(element *x);					// house-keeping after deletion
	dppair    *consSubtree(element *z);					// internal recursive cons'ing function
	dpair	*returnSubtreeAsList(element *z, dpair *head);
	void		printSubTree(element *z);					// display the subtree rooted at z
	void		deleteSubTree(element *z);					// delete subtree rooted at z
	element   *returnMinKey(element *z);					// returns minimum of subtree rooted at z
	element   *returnSuccessor(element *z);					// returns successor of z's key
	
public:
	vektor(int size); ~vektor();							// default constructor/destructor

	element*  findItem(const int searchKey);				// returns T if searchKey found, and
													// points foundNode at the corresponding node
	void		insertItem(int newKey, double newStored);		// insert a new key with stored value
	void		insertItem2(int newKey, double newStored);		// insert a new key with stored value
	void		deleteItem(int killKey);						// selete a node with given key
	void		deleteTree();								// delete the entire tree
	dpair	*returnTreeAsList();						// return the tree as a list of dpairs
	dpair	*returnTreeAsList2();						// return the tree as a list of dpairs
	tuple	returnMaxKey();							// returns the maximum key in the tree
	tuple	returnMaxStored();							// returns a tuple of the maximum (key, .stored)
	int		returnNodecount();							// returns number of items in tree

	void		printTree();								// displays tree (in-order traversal)
	void		printHeap();								// displays heap
	int		returnArraysize();							// 
	int		returnHeaplimit();							// 

};

} // namespace fastcommunity

#endif
