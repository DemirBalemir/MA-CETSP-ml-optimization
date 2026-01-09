    /**
 * Data.cpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#include "Utils/Data.hpp"

Data::Data(Parameters* params) {
    this->instance_index = params->instance_index;
    this->filename = FILENAMES[params->instance_index];
    this->timestamp = params->timestamp;
    this->result_filename = "";

    if (ENV == "LOCAL") {
        data_dir = LOCAL_DATA_DIR;
        res_dir = LOCAL_RES_DIR;
    }
    else if (ENV == "SERVER") {
        data_dir = SERVER_DATA_DIR;
        res_dir = SERVER_RES_DIR;
    }
}

void Data::read() {
    std::cout << "instance : " << filename << std::endl;
    int file_type;
    if (instance_index >= 1 && instance_index <= 20) {
        file_type = 1;
    }
    else {
        file_type = 2;
    }
    try {
        std::string data;
        std::ifstream in;
        in.open(data_dir + filename + ".cetsp");

        // get info of vertices
        while (!in.eof()) {
            getline(in, data);
            if (data.size() > 1) {
                std::istringstream iss(data);
                double x, y, z, r, d;
                iss >> x >> y >> z >> r >> d;
                centers.emplace_back(std::vector<double> {x, y, r});
            }
            else break;
        }
        // get the info of depot
        if (instance_index <= 61) {
            std::vector<double> depot;
            getline(in, data);
            if (file_type == 2) getline(in, data);
            std::istringstream iss(data);
            std::string placeholder, x, y, z;
            if (file_type == 1)
                iss >> placeholder >> placeholder >> x >> y >> z;
            else if (file_type == 2)
                iss >> placeholder >> x >> y >> z;
            depot.emplace_back(std::stof(x));
            depot.emplace_back(std::stof(y));
            depot.emplace_back(0);
            centers.insert(centers.begin(), depot);
        }
        in.close();
    }
    catch (std::exception e) {
        std::cout << "ERROR : read data" << std::endl;
    }
    reduceSize();
}

void Data::write(List* solution, int iter, std::string running_time) {
    try {
        if (!std::filesystem::exists(res_dir)) {
            std::filesystem::create_directory(res_dir);
        }
        std::string running_dir = res_dir + "normal/" + filename + "-" + this->timestamp + "/";
        if (!std::filesystem::exists(running_dir)) {
            std::filesystem::create_directory(running_dir);
        }
        std::ofstream out;
        result_filename = filename + "-" + this->timestamp + "-" + std::to_string(iter);
        if (LOG) {
            std::cout << "file name : " << result_filename << std::endl;
        }
        out.open(running_dir + result_filename + ".txt");
        Node* p = solution->head();
        for (int i = 0; i < solution->size(); ++i) {
            out << p->id << ",";
            p = p->next;
        }
        out << "\n";
        out << "value : " << solution->getValue() << " " << "running time : " << running_time << "\n";
        p = solution->head();
        for (int i = 0; i < solution->size(); ++i) {
            out << p->id << " " << p->x << " " << p->y << "\n";
            p = p->next;
        }
        out.close();
    }
    catch (std::exception e) {
        std::cout << "ERROR : write result" << std::endl;
    }
}
void Data::writeSolutionLog(List* s) {
    try {
        // directory for ML logs
        std::string log_root = res_dir + "ml_logs/";
        if (!std::filesystem::exists(log_root)) {
            std::filesystem::create_directories(log_root);
        }

        // create directory for this instance / timestamp
        std::string logdir = log_root + filename + "-" + timestamp + "/";
        if (!std::filesystem::exists(logdir)) {
            std::filesystem::create_directories(logdir);
        }

        // unique filename for this solution
        static long long counter = 0;
        std::string fname = "sol-" + std::to_string(counter++) + ".json";

        std::ofstream out(logdir + fname);

        out << "{\n";
        out << "  \"instance_index\": " << instance_index << ",\n";
        out << "  \"birth_iter\": " << s->birth_iter << ",\n";
        out << "  \"death_iter\": " << s->death_iter << ",\n";
        out << "  \"survival_iters\": " << (s->death_iter - s->birth_iter) << ",\n";
        out << "  \"censored\": " << (s->censored ? "true" : "false") << ",\n";

        // PRE-VND COST
        out << "  \"pre_vnd_cost\": " << s->pre_vnd_value << ",\n";

        // POST-VND COST
        out << "  \"post_vnd_cost\": " << s->post_vnd_value << ",\n";

        // pre-VND coords
        out << "  \"pre_vnd_coords\": [";
        for (int i = 0; i < s->pre_vnd_coords.size(); ++i) {
            out << "[" << s->pre_vnd_coords[i].first
                << "," << s->pre_vnd_coords[i].second << "]";
            if (i + 1 < s->pre_vnd_coords.size()) out << ", ";
        }
        out << "],\n";

        // FITNESS VALUES
        out << "  \"post_vnd_fitness\": " << s->post_vnd_fitness_at_birth << ",\n";
        out << "  \"final_fitness\": " << s->final_fitness << "\n";

        out << "}\n";

        out.close();
    }
    catch (std::exception e) {
        std::cout << "ERROR : writeSolutionLog" << std::endl;
    }
}




void Data::reduceSize() {
    // reduce the vertices which are covered by depot
    double x0 = centers[0][0], y0 = centers[0][1];
    auto it = centers.begin() + 1;
    while (it != centers.end()) {
        double x = it->at(0), y = it->at(1), r = it->at(2);
        if (pow(x - x0, 2) + pow(y - y0, 2) < pow(r, 2)) {
            it = centers.erase(it);
        }
        else {
            ++it;
        }
    }
    // reduce the centers which are covered by others
    int radius_type;
    if (instance_index >= 48 && instance_index <= 61) {
        radius_type = 2;
    }
    else {
        radius_type = 1;
    }
    if (radius_type == 2) {
        auto iti = centers.begin() + 1;
        while (iti != centers.end()) {
            auto itj = iti + 1;
            while (itj != centers.end()) {
                double x1 = iti->at(0), y1 = iti->at(1), r1 = iti->at(2);
                double x2 = itj->at(0), y2 = itj->at(1), r2 = itj->at(2);
                double dist = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
                if (dist <= r2 - r1) {
                    itj = centers.erase(itj);
                }
                else if (dist <= r1 - r2) {
                    iti = centers.erase(iti);
                    --iti;
                    break;
                }
                else {
                    ++itj;
                }
            }
            ++iti;
        }
    }
}

Centers Data::getData() {
    read();
    return centers;
}

std::string Data::getResultFilename() {
    return result_filename;
}