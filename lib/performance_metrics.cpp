#include "../include/performance_metrics.h"


// FUNCTION MEMBERS
double PerformanceMetrics:: compute_detection_accuracy(){

    // Parser for predicted labels
    parser(this->path_pred_labels, this->sugar_p, this->mustard_p, this->power_drill_p);
    
    // Parser for true labels
    parser(this->path_true_labels, this->sugar_t, this->mustard_t, this->power_drill_t);

    std::cout<<"Predicted labels:\n\n";
    printValue(this->sugar_p, this->mustard_p, this->power_drill_p);

    std::cout<<"True labels:\n\n";
    printValue(this->sugar_t, this->mustard_t, this->power_drill_t);
    
    // Calculus of the metrics

    return 0;

}

// HELPER FUCNTIONS
//std::vector<cv::Point>
void parser(const std::string& path , std::vector<std::pair<int, int>>& sugar, std::vector<std::pair<int, int>>& mustard, std::vector<std::pair<int, int>>& power_drill){

    std::ifstream file(path);

    if(!file.is_open()){
        std::cerr<< "Impossibile to open the file \n";
        return;
    }
    std::string line;

    while(std::getline(file,line)){

        std::vector<std::string> words;
        std::istringstream iss(line);
        std::string word;
        
        // Dividing the line using space as separator
        while(iss >> word) {
            words.push_back(word);
        }

        if(words[0] == "004_sugar_box"){

            sugar[0] = std::make_pair(std::stoi(words[1]), std::stoi(words[2]));
            sugar[1] = std::make_pair(std::stoi(words[3]), std::stoi(words[4]));

        }else if(words[0] == "006_mustard_bottle"){

            mustard[0] = std::make_pair(std::stoi(words[1]), std::stoi(words[2]));
            mustard[1] = std::make_pair(std::stoi(words[3]), std::stoi(words[4]));

        }else if (words[0] == "035_power_drill")
        {
            power_drill[0] = std::make_pair(std::stoi(words[1]), std::stoi(words[2]));
            power_drill[1] = std::make_pair(std::stoi(words[3]), std::stoi(words[4]));
        }
        
        // closed automatically the file open thanks to the distructor of ifstream
        
    }

}

void printValue(std::vector<std::pair<int, int>> v1, std::vector<std::pair<int, int>> v2, std::vector<std::pair<int, int>> v3){

    std::cout << "Sugar Coord:" << std::endl;
    for (const auto& p : v1) {
        std::cout << "(" << p.first << ", " << p.second << ") ";
    }
    std::cout << std::endl;

    std::cout << "Mustard Coord:" << std::endl;
    for (const auto& p : v2) {
        std::cout << "(" << p.first << ", " << p.second << ") ";
    }
    std::cout << std::endl;

    std::cout << "Power Drill Coord:" << std::endl;
    for (const auto& p : v3) {
        std::cout << "(" << p.first << ", " << p.second << ")";
    }
    std::cout << "\n\n";

}
double PerformanceMetrics::compute_mIoU();