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
    
    std::getline(infile,line);
    outfile<<"date, lat, lon, h" <<std::endl;

    while(std::getline(infile,line)){
        
        std::stringstream line_stream(line);

        // PK_UID
        std::getline(line_stream, item, ',');

        // Name
        std::getline(line_stream, item, ',');

        // Photo
        std::getline(line_stream, item, ',');

        //Video
        std::getline(line_stream, item, ',');
        
        //AntH
        std::getline(line_stream, item, ',');

        //Solution
        std::getline(line_stream, item, ',');

        //DATE
        std::getline(line_stream, item, ',');
        time_t t = strTime2unix(item);
        outfile<<t <<"000,";

        //HRMS
        std::getline(line_stream, item, ',');

        //VRMS
        std::getline(line_stream, item, ',');

        //SatNum
        std::getline(line_stream, item, ',');

        //lat
        std::getline(line_stream, item, ',');
        outfile<<item<<",";
        
        //lon
        std::getline(line_stream, item, ',');
        outfile<<item<<",";
        //H
        std::getline(line_stream, item, ',');
        outfile<<item<<std::endl;
        //x
        std::getline(line_stream, item, ',');

        //y
        std::getline(line_stream, item, ',');

        //h
        std::getline(line_stream, item, ',');
    }
    infile.close();
    outfile.close();
}
