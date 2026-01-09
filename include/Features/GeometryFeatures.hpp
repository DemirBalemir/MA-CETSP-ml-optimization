#pragma once
#include <map>
#include <string>
#include <vector>


namespace GeometryFeatures {

	using Coord = std::pair<double, double>;
	using FeatureMap = std::map<std::string, double>;

	FeatureMap extract(const std::vector<Coord>& coords);
	std::string to_json(const FeatureMap& feats);

}
