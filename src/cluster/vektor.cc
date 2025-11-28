#include "vektor.h"

namespace fastcommunity {
// ------------------------------------------------------------------------------------
// Red-Black Tree methods

vektor::vektor(int size) {
	root = new element;
	leaf = new element;
	heap = new maxheap(size);

	leaf->parent   = root;

	root->left	= leaf;
	root->right    = leaf;
	support		= 0;
}

vektor::~vektor() {
	if (root != NULL && (root->left != leaf || root->right != leaf)) { deleteSubTree(root); }
	support         = 0;
	delete leaf;
	root		= NULL;
	leaf		= NULL;
	delete heap;
	heap		= NULL;
}

void vektor::deleteTree() { if (root != NULL) { deleteSubTree(root); } return; }

void vektor::deleteSubTree(element *z) {

	if (z->left  != leaf) { deleteSubTree(z->left);  }
	if (z->right != leaf) { deleteSubTree(z->right); }
	delete z;
	z = NULL;
	return;
}


// Search Functions -------------------------------------------------------------------

// public search function - if there exists a element in the three with key=searchKey,
// it returns TRUE and foundNode is set to point to the found node; otherwise, it sets
// foundNode=NULL and returns FALSE
element* vektor::findItem(const int searchKey) {

	element *current;    current = root;
	if (current->key==0) { return NULL; }							// empty tree; bail out
	while (current != leaf) {
		if (searchKey < current->key) {							// left-or-right?
			if (current->left  != leaf) { current = current->left;  }	// try moving down-left
			else { return NULL; }								//   failure; bail out
		} else {												// 
			if (searchKey > current->key) {							// left-or-right?
				if (current->right  != leaf) { current = current->right;  }	// try moving down-left
				else { return NULL; }							//   failure; bail out
			} else { return current; }							// found (searchKey==current->key)
		}
	}
	return NULL;
}

// Return Item Functions -------------------------------------------------------------------
// public function which returns the tree, via pre-order traversal, as a linked list

dpair* vektor::returnTreeAsList() { // pre-order traversal
	
	dpair  *head, *tail;

	head    = new dpair;
	head->x = root->key;
	head->y = root->stored;
	tail = head;

	if (root->left  != leaf) { tail = returnSubtreeAsList(root->left,  tail); }
	if (root->right != leaf) { tail = returnSubtreeAsList(root->right, tail); }
	
	if (head->x==0) { return NULL; /* empty tree */} else { return head; }
}

dpair* vektor::returnSubtreeAsList(element *z, dpair *head) {
	dpair *newnode, *tail;
	
	newnode    = new dpair;
	newnode->x = z->key;
	newnode->y = z->stored;
	head->next = newnode;
	tail       = newnode;
	
	if (z->left  != leaf) { tail = returnSubtreeAsList(z->left,  tail); }
	if (z->right != leaf) { tail = returnSubtreeAsList(z->right, tail); }
	
	return tail;
}

tuple vektor::returnMaxStored() { return heap->returnMaximum(); }

tuple vektor::returnMaxKey() {
	tuple themax;
	element *current;
	current = root;
	while (current->right != leaf) {		// search to bottom-right corner of tree
		current = current->right; }		// 
	themax.m = current->stored;			// store the data found
	themax.i = current->key;				// 
	themax.j = current->key;				// 
	
	return themax;						// return that data
}

// private functions for deleteItem() (although these could easily be made public, I suppose)
element* vektor::returnMinKey(element *z) {
	element *current;

	current = z;
	while (current->left != leaf) {		// search to bottom-right corner of tree
		current = current->left; }		// 
	return current;					// return pointer to the minimum
}

element* vektor::returnSuccessor(element *z) {
	element *current, *w;
	
	w = z;
	if (w->right != leaf) {				// if right-subtree exists, return min of it
		return returnMinKey(w->right); }
	current = w->parent;				// else search up in tree
	while ((current!=NULL) && (w==current->right)) {
		w       = current;
		current = current->parent;		// move up in tree until find a non-right-child
	}
	return current;
}

int vektor::returnNodecount() { return support; }
int vektor::returnArraysize() { return heap->returnArraysize(); }
int vektor::returnHeaplimit() { return heap->returnHeaplimit(); }

// Heapification Functions -------------------------------------------------------------------

// Insert Functions -------------------------------------------------------------------
// public insert function
void vektor::insertItem(int newKey, double newStored) {
	
	// first we check to see if newKey is already present in the tree; if so, we simply
	// set .stored += newStored; if not, we must find where to insert the key
	element *newNode, *current;
	bool foundFlag;

	current = findItem(newKey);						// find newKey in tree; return pointer to it O(log k)
	if (current != NULL) {
		current->stored += newStored;					// update its stored value
		heap->updateItem(current->heap_ptr, current->stored);
												// update corresponding element in heap + reheapify; O(log k)
	} else {										// didn't find it, so need to create it
		tuple newitem;								// 
		newitem.m = newStored;						//  
		newitem.i = -1;							//  
		newitem.j = newKey;							//  
		
		newNode			= new element;				// element for the vektor
		newNode->key		= newKey;					//  store newKey
		newNode->stored	= newStored;  				//  store newStored
		newNode->color		= true;					//  new nodes are always RED
		newNode->parent	= NULL;					//  new node initially has no parent
		newNode->left		= leaf;					//  left leaf
		newNode->right		= leaf;					//  right leaf
		newNode->heap_ptr   = heap->insertItem(newitem);  // add new item to the vektor heap
		support++;								// increment node count in vektor

		// must now search for where to insert newNode, i.e., find the correct parent and
		// set the parent and child to point to each other properly
		current = root;
		if (current->key==0) {										// insert as root
			delete root;											//   delete old root
			root			= newNode;								//   set root to newNode
			leaf->parent   = newNode;								//   set leaf's parent
			current		= leaf;									//   skip next loop
		}
		
		while (current != leaf) {									// search for insertion point
			if (newKey < current->key) {								// left-or-right?
				if (current->left  != leaf) { current = current->left;  }	// try moving down-left
				else {											// else found new parent
					newNode->parent	= current;					//    set parent
					current->left		= newNode;					//    set child
					current			= leaf;						//    exit search
				}
			} else {												// 
				if (current->right != leaf) { current = current->right; }   // try moving down-right
				else {											// else found new parent
					newNode->parent	= current;					//    set parent
					current->right		= newNode;					//    set child
					current			= leaf;						//    exit search
				}
			}
		}

		// now do the house-keeping necessary to preserve the red-black properties
		insertCleanup(newNode);			// do house-keeping to maintain balance
	}
	return;
}

// public insert function
void vektor::insertItem2(int newKey, double newStored) {
	
	// first we check to see if newKey is already present in the tree; if so, we simply
	// set .stored += newStored; if not, we must find where to insert the key
	element *newNode, *current;
	bool foundFlag;

	current = findItem(newKey);						// find newKey in tree; return pointer to it O(log k)
	if (current != NULL) {
		current->stored = newStored;					// update its stored value
		heap->updateItem(current->heap_ptr, current->stored);
												// update corresponding element in heap + reheapify; O(log k)
	} else {										// didn't find it, so need to create it
		tuple newitem;								// 
		newitem.m = newStored;						//  
		newitem.i = -1;							//  
		newitem.j = newKey;							//  
		
		newNode			= new element;				// element for the vektor
		newNode->key		= newKey;					//  store newKey
		newNode->stored	= newStored;  				//  store newStored
		newNode->color		= true;					//  new nodes are always RED
		newNode->parent	= NULL;					//  new node initially has no parent
		newNode->left		= leaf;					//  left leaf
		newNode->right		= leaf;					//  right leaf
		newNode->heap_ptr   = heap->insertItem(newitem);  // add new item to the vektor heap
		support++;								// increment node count in vektor

		// must now search for where to insert newNode, i.e., find the correct parent and
		// set the parent and child to point to each other properly
		current = root;
		if (current->key==0) {										// insert as root
			delete root;											//   delete old root
			root			= newNode;								//   set root to newNode
			leaf->parent   = newNode;								//   set leaf's parent
			current		= leaf;									//   skip next loop
		}
		
		while (current != leaf) {									// search for insertion point
			if (newKey < current->key) {								// left-or-right?
				if (current->left  != leaf) { current = current->left;  }	// try moving down-left
				else {											// else found new parent
					newNode->parent	= current;					//    set parent
					current->left		= newNode;					//    set child
					current			= leaf;						//    exit search
				}
			} else {												// 
				if (current->right != leaf) { current = current->right; }   // try moving down-right
				else {											// else found new parent
					newNode->parent	= current;					//    set parent
					current->right		= newNode;					//    set child
					current			= leaf;						//    exit search
				}
			}
		}

		// now do the house-keeping necessary to preserve the red-black properties
		insertCleanup(newNode);			// do house-keeping to maintain balance
	}
	return;
}

// private house-keeping function for insertion
void vektor::insertCleanup(element *z) {
	
	if (z->parent==NULL) {								// fix now if z is root
		z->color = false; return; }
	element *temp;
	while (z->parent!=NULL && z->parent->color) {	// while z is not root and z's parent is RED
		if (z->parent == z->parent->parent->left) {  // z's parent is LEFT-CHILD
			temp = z->parent->parent->right;		// grab z's uncle
			if (temp->color) {
				z->parent->color		= false;  // color z's parent BLACK	(Case 1)
				temp->color			= false;  // color z's uncle BLACK		(Case 1)
				z->parent->parent->color = true;   // color z's grandparent RED  (Case 1)
				z = z->parent->parent;			// set z = z's grandparent    (Case 1)
			} else {
				if (z == z->parent->right) {		// z is RIGHT-CHILD
					z = z->parent;				// set z = z's parent		(Case 2)
					rotateLeft(z);				// perform left-rotation		(Case 2)
				}
				z->parent->color		= false;  // color z's parent BLACK	(Case 3)
				z->parent->parent->color = true;   // color z's grandparent RED  (Case 3)
				rotateRight(z->parent->parent);    // perform right-rotation	(Case 3)
			}
		} else {								// z's parent is RIGHT-CHILD
			temp = z->parent->parent->left;		// grab z's uncle
			if (temp->color) {
				z->parent->color		= false;  // color z's parent BLACK	(Case 1)
				temp->color			= false;  // color z's uncle BLACK		(Case 1)
				z->parent->parent->color = true;   // color z's grandparent RED  (Case 1)
				z = z->parent->parent;			// set z = z's grandparent    (Case 1)
			} else {
				if (z == z->parent->left) {		// z is LEFT-CHILD
					z = z->parent;				// set z = z's parent		(Case 2)
					rotateRight(z);			// perform right-rotation	(Case 2)
				}
				z->parent->color		= false;  // color z's parent BLACK	(Case 3)
				z->parent->parent->color = true;   // color z's grandparent RED  (Case 3)
				rotateLeft(z->parent->parent);	// perform left-rotation		(Case 3)
			}
		}
	}

	root->color = false;						// color the root BLACK
	return;
}

// Delete Functions -------------------------------------------------------------------
// public delete function
void vektor::deleteItem(int killKey) {
	element *x, *y, *z;
	char pauseme;
	
	z = findItem(killKey);
	if (z == NULL) { return; }						// item not present; bail out

	if (z != NULL) {
		tuple newmax    = heap->returnMaximum();		// get old maximum in O(1)
		heap->deleteItem(z->heap_ptr);				// delete item in the max-heap O(log k)
	}
	
	if (support==1) {								// -- attempt to delete the root
		root->key		= 0;							// restore root node to default state
		root->stored   = -4294967296.0;				// 
		root->color    = false;						// 
		root->parent   = NULL;						// 
		root->left	= leaf;						// 
		root->right    = leaf;						// 
		root->heap_ptr = NULL;						// 
		support--;								// set support to zero
		return;									// exit - no more work to do
	}
	
	if (z != NULL) {
		support--;								// decrement node count
		if ((z->left == leaf) || (z->right==leaf)) {		// case of less than two children
			  y = z; }							//    set y to be z
		else { y = returnSuccessor(z); }				//    set y to be z's key-successor
		
		if (y->left!=leaf) { x = y->left; }			// pick y's one child (left-child)
		else			    { x = y->right; }			//				  (right-child)
		x->parent = y->parent;						// make y's child's parent be y's parent

		if (y->parent==NULL) { root = x; }				// if y is the root, x is now root
		else {									// 
			if (y == y->parent->left) {				// decide y's relationship with y's parent
				y->parent->left  = x;				//   replace x as y's parent's left child
			} else {								// 
				y->parent->right = x; }				//   replace x as y's parent's left child
		}										// 

		if (y!=z) {								// insert y into z's spot
			z->key		= y->key;					// copy y data into z
			z->stored		= y->stored;				// 
			z->heap_ptr    = y->heap_ptr;				// 
		}										// 

		if (y->color==false) { deleteCleanup(x); }		// do house-keeping to maintain balance
		delete y;									// deallocate y
		y = NULL;									// point y to NULL for safety
	}											// 
		
	return;
}

void vektor::deleteCleanup(element *x) {
	element *w, *t;
	while ((x != root) && (x->color==false)) {			// until x is the root, or x is RED
		if (x==x->parent->left) {					// branch on x being a LEFT-CHILD
			w = x->parent->right;					// grab x's sibling
			if (w->color==true) {					// if x's sibling is RED
				w->color = false;					// color w BLACK				(case 1)
				x->parent->color = true;				// color x's parent RED			(case 1)
				rotateLeft(x->parent);				// left rotation on x's parent	(case 1)
				w = x->parent->right;				// make w be x's right sibling	(case 1)
			}
			if ((w->left->color==false) && (w->right->color==false)) {
				w->color = true;					// color w RED					(case 2)
				x = x->parent;						// examine x's parent			(case 2)
			} else {								// 
				if (w->right->color==false) {			// 
					w->left->color = false;			// color w's left child BLACK		(case 3)
					w->color = true;				// color w RED					(case 3)
					t = x->parent;					// store x's parent
					rotateRight(w);				// right rotation on w			(case 3)
					x->parent = t;					// restore x's parent
					w = x->parent->right;			// make w be x's right sibling	(case 3)
				}								// 
				w->color			= x->parent->color; // make w's color = x's parent's   (case 4)
				x->parent->color    = false;			// color x's parent BLACK		(case 4)
				w->right->color	= false;			// color w's right child BLACK	(case 4)
				rotateLeft(x->parent);				// left rotation on x's parent	(case 4)
				x = root;							// finished work. bail out		(case 4)
			}									// 
		} else {									// x is RIGHT-CHILD
			w = x->parent->left;					// grab x's sibling
			if (w->color==true) {					// if x's sibling is RED
				w->color			= false;			// color w BLACK				(case 1)
				x->parent->color    = true;			// color x's parent RED			(case 1)
				rotateRight(x->parent);				// right rotation on x's parent	(case 1)
				w				= x->parent->left;  // make w be x's left sibling		(case 1)
			}
			if ((w->right->color==false) && (w->left->color==false)) {
				w->color = true;					// color w RED					(case 2)
				x= x->parent;						// examine x's parent			(case 2)
			} else {								// 
				if (w->left->color==false) {			// 
					w->right->color	= false;		// color w's right child BLACK	(case 3)
					w->color			= true;		// color w RED					(case 3)
					t				= x->parent;   // store x's parent
					rotateLeft(w);					// left rotation on w			(case 3)
					x->parent			= t;			// restore x's parent
					w = x->parent->left;			// make w be x's left sibling		(case 3)
				}								// 
				w->color = x->parent->color;			// make w's color = x's parent's   (case 4)
				x->parent->color    = false;			// color x's parent BLACK		(case 4)
				w->left->color		= false;			// color w's left child BLACK		(case 4)
				rotateRight(x->parent);				// right rotation on x's parent    (case 4)
				x				= root;			// x is now the root			(case 4)
			}
		}
	}
	x->color = false;								// color x (the root) BLACK		(exit)

	return;
}

// Rotation Functions -------------------------------------------------------------------

void vektor::rotateLeft(element *x) {
	element *y;
	// do pointer-swapping operations for left-rotation
	y               = x->right;					// grab right child
	x->right        = y->left;					// make x's RIGHT-CHILD be y's LEFT-CHILD
	y->left->parent = x;						// make x be y's LEFT-CHILD's parent
	y->parent       = x->parent;					// make y's new parent be x's old parent

	if (x->parent==NULL) { root = y; }				// if x was root, make y root
	else {									// 
		if (x == x->parent->left)				// if x is LEFT-CHILD, make y be x's parent's
			{ x->parent->left  = y; }			//    left-child
		else { x->parent->right = y; }			//    right-child
	}										// 
	y->left   = x;								// make x be y's LEFT-CHILD
	x->parent = y;								// make y be x's parent
	
	return;
}

void vektor::rotateRight(element *y) {
	element *x;
	// do pointer-swapping operations for right-rotation
	x                = y->left;					// grab left child
	y->left          = x->right;					// replace left child yith x's right subtree
	x->right->parent = y;						// replace y as x's right subtree's parent
	
	x->parent        = y->parent;					// make x's new parent be y's old parent
	if (y->parent==NULL) { root = x; }				// if y was root, make x root
	else {
		if (y == y->parent->right)				// if y is RIGHT-CHILD, make x be y's parent's
			{ y->parent->right  = x; }			//    right-child
		else { y->parent->left   = x; }			//    left-child
	}
	x->right  = y;								// make y be x's RIGHT-CHILD
	y->parent = x;								// make x be y's parent
	
	return;
}

// Display Function -------------------------------------------------------------------
// public
void vektor::printTree() {
	tuple max = heap->returnMaximum();
	std::cout << "\nTREE SIZE = " << support << std::endl;
	std::cout << "HEAP SIZE = " << heap->heapSize() << std::endl;
	std::cout << "MAXIMUM (" << max.j<< " " << max.m << ")" << std::endl;
	std::cout << "# "; printSubTree(root);
	return;
}

void vektor::printHeap() { heap->printHeap(); return; }

//private
void vektor::printSubTree(element *z) {
	if (z==leaf) { return; }
	else {
		std::cout << "(" << z->key << " " << z->stored << " " << z->color << ") [" << z->heap_ptr << "]"<< std::endl;
		std::cout << "L "; printSubTree(z->left); std::cout << std::endl;
		std::cout << "R "; printSubTree(z->right); std::cout << std::endl;
	}
	return;
}
// ------------------------------------------------------------------------------------
}