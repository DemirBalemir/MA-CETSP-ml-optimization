#include <algorithm>
#include <cmath>
#include <sstream>
#include "GeometryFeatures.hpp"


namespace GeometryFeatures {

    FeatureMap extract(const std::vector<Coord>& coords)
    {
        FeatureMap F;

        int n = coords.size();
        if (n < 2) {
            // fallback
            F["avg_edge_length"] = 0;
            F["var_edge_length"] = 0;
            F["bbox_width"] = 0;
            F["bbox_height"] = 0;
            F["bbox_area"] = 0;
            F["centroid_x"] = 0;
            F["centroid_y"] = 0;
            F["centroid_dist_sum"] = 0;
            F["angle_variance"] = 0;
            return F;
        }

        std::vector<double> xs(n), ys(n);
        for (int i = 0; i < n; i++) {
            xs[i] = coords[i].first;
            ys[i] = coords[i].second;
        }

        // ------------------------
        // EDGE LENGTHS
        // ------------------------
        std::vector<double> edges;
        edges.reserve(n);

        for (int i = 0; i < n - 1; i++) {
            double dx = xs[i + 1] - xs[i];
            double dy = ys[i + 1] - ys[i];
            edges.push_back(std::sqrt(dx * dx + dy * dy));
        }

        double sum_e = 0;
        for (double e : edges) sum_e += e;
        double avg_e = sum_e / edges.size();

        double var_sum = 0;
        for (double e : edges) var_sum += (e - avg_e) * (e - avg_e);
        double var_e = var_sum / edges.size();

        F["avg_edge_length"] = avg_e;
        F["var_edge_length"] = var_e;

        // ------------------------
        // BOUNDING BOX
        // ------------------------
        double min_x = *std::min_element(xs.begin(), xs.end());
        double max_x = *std::max_element(xs.begin(), xs.end());
        double min_y = *std::min_element(ys.begin(), ys.end());
        double max_y = *std::max_element(ys.begin(), ys.end());

        double width = max_x - min_x;
        double height = max_y - min_y;

        F["bbox_width"] = width;
        F["bbox_height"] = height;
        F["bbox_area"] = width * height;

        // ------------------------
        // CENTROID
        // ------------------------
        double cx = 0, cy = 0;
        for (int i = 0; i < n; i++) {
            cx += xs[i];
            cy += ys[i];
        }
        cx /= n;
        cy /= n;

        F["centroid_x"] = cx;
        F["centroid_y"] = cy;

        // ------------------------
        // DISTANCE TO CENTROID
        // ------------------------
        double dist_sum = 0;
        for (int i = 0; i < n; i++) {
            double dx = xs[i] - cx;
            double dy = ys[i] - cy;
            dist_sum += std::sqrt(dx * dx + dy * dy);
        }
        F["centroid_dist_sum"] = dist_sum;

        // ------------------------
        // ANGLE VARIANCE (smoothness)
        // ------------------------
        std::vector<double> angles;
        for (int i = 1; i < n - 1; i++) {
            double ax = xs[i - 1] - xs[i];
            double ay = ys[i - 1] - ys[i];
            double bx = xs[i + 1] - xs[i];
            double by = ys[i + 1] - ys[i];

            double dot = ax * bx + ay * by;
            double normA = std::sqrt(ax * ax + ay * ay);
            double normB = std::sqrt(bx * bx + by * by);

            if (normA < 1e-9 || normB < 1e-9) continue;

            double cosang = dot / (normA * normB);
            if (cosang > 1) cosang = 1;
            if (cosang < -1) cosang = -1;

            angles.push_back(std::acos(cosang));
        }

        double var_ang = 0;
        if (!angles.empty()) {
            double avg_ang = 0;
            for (double a : angles) avg_ang += a;
            avg_ang /= angles.size();

            double vap = 0;
            for (double a : angles) vap += (a - avg_ang) * (a - avg_ang);
            var_ang = vap / angles.size();
        }

        F["angle_variance"] = var_ang;

        return F;
    }

    std::string to_json(const FeatureMap& feats)
    {
        std::ostringstream oss;
        oss << "{";
        bool first = true;

        for (const auto& kv : feats) {
            if (!first) oss << ",";
            first = false;
            oss << "\"" << kv.first << "\":" << kv.second;
        }

        oss << "}";
        return oss.str();
    }

} // namespace GeometryFeatures
