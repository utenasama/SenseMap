//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"
#include "util/math.h"

#include <cstdarg>

#include <boost/algorithm/string.hpp>

namespace sensemap {

std::string EnsureTrailingSlash(const std::string& str) {
	if (str.length() > 0) {
		if (str.back() != '/') {
			return str + "/";
		}
	} else {
		return str + "/";
	}
	return str;
}

bool IsInsideSubpath(const std::string& path, const std::string& sub_path) {
	std::string sub_path2 = EnsureTrailingSlash(sub_path);
	return path.substr(0, sub_path2.length()) == sub_path2;
}

bool HasFileExtension(const std::string& file_name, const std::string& ext) {
	CHECK(!ext.empty());
	CHECK_EQ(ext.at(0), '.');
	std::string ext_lower = ext;
	StringToLower(&ext_lower);
	if (file_name.size() >= ext_lower.size() &&
	    file_name.substr(file_name.size() - ext_lower.size(), ext_lower.size()) ==
	    ext_lower) {
		return true;
	}
	return false;
}

void SplitFileExtension(const std::string& path, std::string* root,
                        std::string* ext) {
	const auto parts = StringSplit(path, ".");
	CHECK_GT(parts.size(), 0);
	if (parts.size() == 1) {
		*root = parts[0];
		*ext = "";
	} else {
		*root = "";
		for (size_t i = 0; i < parts.size() - 1; ++i) {
			*root += parts[i] + ".";
		}
		*root = root->substr(0, root->length() - 1);
		if (parts.back() == "") {
			*ext = "";
		} else {
			*ext = "." + parts.back();
		}
	}
}

bool ExistsFile(const std::string& path) {
	return boost::filesystem::is_regular_file(path);
}

bool ExistsDir(const std::string& path) {
	return boost::filesystem::is_directory(path);
}

bool ExistsPath(const std::string& path) {
	return boost::filesystem::exists(path);
}

void CreateDirIfNotExists(const std::string& path) {
	if (!ExistsDir(path)) {
		CHECK(boost::filesystem::create_directory(path));
	}
}

std::string GetPathBaseName(const std::string& path) {
	const std::vector<std::string> names =
			StringSplit(StringReplace(path, "\\", "/"), "/");
	if (names.size() > 1 && names.back() == "") {
		return names[names.size() - 2];
	} else {
		return names.back();
	}
}

std::string GetVideoTimeStampFromPath(const std::string& path) {
	const std::vector<std::string> names =
			StringSplit(StringReplace(path, "\\", "/"), "/");
	for (const std::string& name : names) {
		const std::vector<std::string> elems = StringSplit(name, "_");
		if (elems.size() < 2) {
			continue;
		}

		for (auto elem : elems) {
			std::cout << " " << elem;
		}
		std::cout << "\n";

		for (int i = 1; i < elems.size(); i++) {
			if (!IsDouble(elems[i-1]) || !IsDouble(elems[i])){
				continue;
			}

			if ((elems[i-1].size() == 8 && elems[i].size() == 6) || (elems[i-1].size() == 8 && elems[i].size() == 9)) {
				int year, month, day, hour, minute, second;
				year = std::stoi(elems[i-1].substr(0, 4));
				// std::cout << "split_elems[1].substr(4, 2) = " << split_elems[1].substr(4, 2) << std::endl;
				month = std::stoi(elems[i-1].substr(4, 2));
				// std::cout << "split_elems[1].substr(6, 2) = " << split_elems[1].substr(6, 2) << std::endl;
				day = std::stoi(elems[i-1].substr(6, 2));
				// std::cout << "split_elems[2].substr(0, 2) = " << split_elems[2].substr(0, 2) << std::endl;
				hour = std::stoi(elems[i].substr(0, 2));
				// std::cout << "split_elems[2].substr(2, 2) = " << split_elems[2].substr(2, 2) << std::endl;
				minute = std::stoi(elems[i].substr(2, 2));
				// std::cout << "split_elems[2].substr(4, 2) = " << split_elems[2].substr(4, 2) << std::endl;
				second = std::stoi(elems[i].substr(4, 2));

				// Check Date and Time Legal or not
				if (IsDateLegal(year, month, day) && IsTimeLegal(hour, minute, second)) {
					struct tm timeinfo;
					timeinfo.tm_year = year - 1900;
					timeinfo.tm_mon = month - 1;
					timeinfo.tm_mday = day;
					timeinfo.tm_hour = hour;
					timeinfo.tm_min = minute;
					timeinfo.tm_sec = second;
					timeinfo.tm_isdst = 0;
					time_t t = mktime(&timeinfo);
					return std::to_string(t) + "000";
				}
				
			} else {
				std::cout << "size wrong" << std::endl;
			}
		}
	}
    
	// Return default string
	return "1609430400000";  //  20210101_000000
}

std::string GetImageTimeStampFromPath(const std::string& path) {
	std::string file_name = GetPathBaseName(path);
	const auto parts = StringSplit(file_name, ".");
	CHECK_GT(parts.size(), 1);

	if (IsDouble(parts[0])){
		return parts[0];
	}

	// Return default string
	return "1609430400000";  //  20210101_000000
}

std::string GetParentDir(const std::string& path) {
	return boost::filesystem::path(path).parent_path().string();
}

std::string GetRelativePath(const std::string& from, const std::string& to) {
	// This implementation is adapted from:
	// https://stackoverflow.com/questions/10167382
	// A native implementation in boost::filesystem is only available starting
	// from boost version 1.60.
	using namespace boost::filesystem;

	path from_path = canonical(path(from));
	path to_path = canonical(path(to));

	// Start at the root path and while they are the same then do nothing then
	// when they first diverge take the entire from path, swap it with '..'
	// segments, and then append the remainder of the to path.
	path::const_iterator from_iter = from_path.begin();
	path::const_iterator to_iter = to_path.begin();

	// Loop through both while they are the same to find nearest common directory
	while (from_iter != from_path.end() && to_iter != to_path.end() &&
	       (*to_iter) == (*from_iter)) {
		++ to_iter;
		++ from_iter;
	}

	// Replace from path segments with '..' (from => nearest common directory)
	path rel_path;
	while (from_iter != from_path.end()) {
		rel_path /= "..";
		++ from_iter;
	}

	// Append the remainder of the to path (nearest common directory => to)
	while (to_iter != to_path.end()) {
		rel_path /= *to_iter;
		++ to_iter;
	}

	return rel_path.string();
}

std::vector<std::string> GetFileList(const std::string& path) {
	std::vector<std::string> file_list;
	for (auto it = boost::filesystem::directory_iterator(path);
	     it != boost::filesystem::directory_iterator(); ++it) {
		if (boost::filesystem::is_regular_file(*it)) {
			const boost::filesystem::path file_path = *it;
			file_list.push_back(file_path.string());
		}
	}
	return file_list;
}

std::vector<std::string> GetRecursiveFileList(const std::string& path) {
	std::vector<std::string> file_list;
	for (auto it = boost::filesystem::recursive_directory_iterator(path);
	     it != boost::filesystem::recursive_directory_iterator(); ++it) {
		if (boost::filesystem::is_regular_file(*it)) {
			const boost::filesystem::path file_path = *it;
			if (file_path.string().substr(file_path.string().length() - 4, file_path.string().length()) != ".txt"){
				file_list.push_back(file_path.string());	
			}
		}
	}
	return file_list;
}

std::vector<std::string> GetDirList(const std::string& path) {
	std::vector<std::string> dir_list;
	for (auto it = boost::filesystem::directory_iterator(path);
	     it != boost::filesystem::directory_iterator(); ++it) {
		if (boost::filesystem::is_directory(*it)) {
			const boost::filesystem::path dir_path = *it;
			dir_list.push_back(dir_path.string());
		}
	}
	return dir_list;
}

std::vector<std::string> GetRecursiveDirList(const std::string& path) {
	std::vector<std::string> dir_list;
	for (auto it = boost::filesystem::recursive_directory_iterator(path);
	     it != boost::filesystem::recursive_directory_iterator(); ++it) {
		if (boost::filesystem::is_directory(*it)) {
			const boost::filesystem::path dir_path = *it;
			dir_list.push_back(dir_path.string());
		}
	}
	return dir_list;
}

size_t GetFileSize(const std::string& path) {
	std::ifstream file(path, std::ifstream::ate | std::ifstream::binary);
	CHECK(file.is_open()) << path;
	return file.tellg();
}

void PrintHeading(const std::string& heading) {
	std::cout << "|" << std::string(heading.size(), '=') << "|" << std::endl;
	std::cout << "|" << heading << "|" << std::endl;
	std::cout << "|" << std::string(heading.size(), '=') << "|" << std::endl;
}

void PrintHeading1(const std::string& heading) {
	std::cout << std::endl << std::string(78, '=') << std::endl;
	std::cout << heading << std::endl;
	std::cout << std::string(78, '=') << std::endl << std::endl;
}

void PrintHeading2(const std::string& heading) {
	std::cout << std::endl << heading << std::endl;
	std::cout << std::string(std::min<int>(heading.size(), 78), '-') << std::endl;
}

void PrintHeading3(const std::string& heading) {
	std::cout << std::endl << heading << std::endl;
}

template <>
std::vector<std::string> CSVToVector(const std::string& csv) {
	auto elems = StringSplit(csv, ",;");
	std::vector<std::string> values;
	values.reserve(elems.size());
	for (auto& elem : elems) {
		StringTrim(&elem);
		if (elem.empty()) {
			continue;
		}
		values.push_back(elem);
	}
	return values;
}

template <>
std::vector<int> CSVToVector(const std::string& csv) {
	auto elems = StringSplit(csv, ",;");
	std::vector<int> values;
	values.reserve(elems.size());
	for (auto& elem : elems) {
		StringTrim(&elem);
		if (elem.empty()) {
			continue;
		}
		try {
			values.push_back(std::stoi(elem));
		} catch (std::exception) {
			return std::vector<int>(0);
		}
	}
	return values;
}

template <>
std::vector<float> CSVToVector(const std::string& csv) {
	auto elems = StringSplit(csv, ",;");
	std::vector<float> values;
	values.reserve(elems.size());
	for (auto& elem : elems) {
		StringTrim(&elem);
		if (elem.empty()) {
			continue;
		}
		try {
			values.push_back(std::stod(elem));
		} catch (std::exception) {
			return std::vector<float>(0);
		}
	}
	return values;
}

template <>
std::vector<double> CSVToVector(const std::string& csv) {
	auto elems = StringSplit(csv, ",;");
	std::vector<double> values;
	values.reserve(elems.size());
	for (auto& elem : elems) {
		StringTrim(&elem);
		if (elem.empty()) {
			continue;
		}
		try {
			values.push_back(std::stold(elem));
		} catch (std::exception) {
			return std::vector<double>(0);
		}
	}
	return values;
}

std::vector<std::string> ReadTextFileLines(const std::string& path) {
	std::ifstream file(path);
	CHECK(file.is_open()) << path;

	std::string line;
	std::vector<std::string> lines;
	while (std::getline(file, line)) {
		StringTrim(&line);

		if (line.empty()) {
			continue;
		}

		lines.push_back(line);
	}

	return lines;
}

void RemoveCommandLineArgument(const std::string& arg, int* argc, char** argv) {
	for (int i = 0; i < *argc; ++i) {
		if (argv[i] == arg) {
			for (int j = i + 1; j < *argc; ++j) {
				argv[i] = argv[j];
			}
			*argc -= 1;
			break;
		}
	}
}

}  // namespace sensemap


