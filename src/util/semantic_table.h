//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_SEMANTIC_TABLE_H_
#define SENSEMAP_UTIL_SEMANTIC_TABLE_H_

#define UNLABEL          0
#define LABEL_WALL       1
#define LABEL_BUILDING   2
#define LABEL_SKY        3
#define LABLE_GROUND     4 // floor, or street
#define LABLE_CEILING    6
#define LABEL_OBJECT     8
#define LABEL_PEDESTRIAN 15
#define LABEL_PLANT      18
#define LABEL_STAIR      19
#define LABEL_OTHER      21
#define LABEL_RIVER      25
#define LABEL_LAMP       37
#define LABEL_POWERLINE  51
#define LABEL_LIGHT      137

namespace sensemap {

extern unsigned char adepallete[765];

extern void LoadSemanticColorTable(const char* filepath);
}

#endif
