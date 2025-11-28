#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <ctime>

time_t strTime2unix(std::string timeStamp)  
{  
    struct tm tm;  
    memset(&tm, 0, sizeof(tm));  
      
    sscanf(timeStamp.c_str(), "%d-%d-%d %d:%d:%d",   
           &tm.tm_year, &tm.tm_mon, &tm.tm_mday,  
           &tm.tm_hour, &tm.tm_min, &tm.tm_sec);  
  
    tm.tm_year -= 1900;  
    tm.tm_mon--;  
  
    return mktime(&tm);  
}

int main(int argc, char* argv[]){

    std::string file_path = argv[1];
    std::string file_path_out = argv[2];


    if (file_path.empty()) return -1;

    // open file
    std::ifstream infile;
    infile.open(file_path.c_str());
    if (!infile.is_open()) {
        return -1;
    }

    std::ofstream outfile;
    outfile.open(file_path_out.c_str());
    if(!outfile.is_open()){
        return -1;
    }

    std::string line;
    std::string item;
    
    //std::getline(infile,line);
    outfile<<"date, lat, lon, h" <<std::endl;

    while(std::getline(infile,line)){
        
        std::stringstream line_stream(line);

        // SLAM_time
        std::getline(line_stream, item, ',');

        // systemtime
        std::getline(line_stream, item, ',');

        outfile<<item<<",";

        //lon
        std::getline(line_stream, item, ',');
        std::string longtitude = item;        
        //lat
        std::getline(line_stream, item, ',');
        std::string latitude = item;
        
        outfile<<latitude<<","<<longtitude<<",";
        //H
        std::getline(line_stream, item, ',');
        outfile<<item<<std::endl;
        //x
        std::getline(line_stream, item, ',');

        //y
        std::getline(line_stream, item, ',');
    }
    infile.close();
    outfile.close();
}
