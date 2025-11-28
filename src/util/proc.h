//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_PROC_H_
#define SENSEMAP_PROC_H_

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#ifndef __linux__
#include <window.h>
#else
#include <unistd.h>
#endif

namespace sensemap {

static bool GetAvailableMemory(float &max_ram){
  char line[50];
  const char delim[2] = " ";
  char *word;
  int mem = 0;
  uint64_t G_byte = 1.0e6;

  max_ram = -1;
  FILE* in_file = fopen("/proc/meminfo","r");
  if(!in_file) exit(-1);
  int line_num=0;
  while(fgets(line,50,in_file) != 0) {
    if(line_num == 2 ) {
        word = strtok(line, delim);
        while(word != NULL) {
            word = strtok(NULL, delim);
            mem = atoi(word);
            max_ram = (float)mem / G_byte;
            // printf("MemAvailable : %f\n",max_ram);
            return true;
        }
    }
    line_num++;
    if(line_num==3){
      printf("False get MemAvailable!\n");
      return false;
    }
  }
  return true;
}

static int GetProcessId() {
  return getpid();
}

}  // namespace sensemap

#endif