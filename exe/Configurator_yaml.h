
#ifndef _CONFIGURATOR_H_
#define _CONFIGURATOR_H_

#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include "yaml-cpp/yaml.h"


class Configurator {
public:
    inline std::string GetArgument(const std::string &directive, const std::string default_param = "") const {
        std::string param = default_param;
        if (node[directive]) {
            param = node[directive].as<std::string>();
        }
        return param;
    }

    inline int GetArgument(const std::string &directive, const int default_param) const {
        int param = default_param;
        if (node[directive]) {
            param = node[directive].as<int>();
        }
        return param;
    }

    inline float GetArgument(const std::string &directive, const float default_param) const {
        float param = default_param;
        if (node[directive]) {
            param = node[directive].as<float>();
        }
        return param;
    }

    inline bool SetArgument(const std::string &directive, const std::string value = "") {
        node[directive] = std::string(value);
        return true;
    }

    inline bool SetArgument(const std::string &directive, const int value) {
        node[directive] = value;
        return true;
    }

    inline bool SetArgument(const std::string &directive, const float value) {
        node[directive] = value;
        return true;
    }

    inline bool Load(const char *fileName) {
        node = YAML::LoadFile(std::string(fileName));
        printf("Loaded \'%s\'\n", fileName);
        return true;
    }

    inline bool Save(const char *fileName) {
        std::ofstream file;
        file.open(std::string(fileName));
        // file.flush();
        file << node;
        file.close();
        return true;
    }

private:
    YAML::Node node;
};

#endif
