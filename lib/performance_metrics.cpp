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

    // IoU =  Area of overlap / Area of union
    // mIoU = (IoU1 +  IoU2 + IoU3) / 3

    // Initialize 2 vectors for the rectangle of our Algo prediction, and for the true label of the dataset
    // In each cell of each vector we find the coordinates of the top left and bottom right corners
    std::vector<cv::Rect> rect_p(3) , rect_t(3);
    rect_p[0] = cv::Rect(sugar_p[0], sugar_p[1]);
    rect_t[0] = cv::Rect(sugar_t[0], sugar_t[1]);

    rect_p[1] = cv::Rect(mustard_p[0], mustard_p[1]);
    rect_t[1] = cv::Rect(mustard_t[0], mustard_t[1]);

    rect_p[2] = cv::Rect(power_drill_p[0], power_drill_p[1]);
    rect_t[2] = cv::Rect(power_drill_t[0], power_drill_t[1]);

    // Define the variable to store the areas of intersection, union and the respective IoU
    double areas_int[3];
    double areas_union[3];
    double IoU[3];

    for (int i = 0; i < rect_p.size(); i++){
        cv::Rect intersect = rect_p[i] & rect_t[i];
        areas_int[i] = intersect.area();
        std::cout<<"Intersection A:" << areas_int[i] << std::endl; 

        areas_union[i] = rect_p[i].area() + rect_t[i].area() - areas_int[i];
        std::cout<<"Union A:" << areas_union[i] << std::endl;

        IoU[i] = areas_int[i] / areas_union[i];
        std::cout<<"IoU :" << IoU[i] << std::endl; 
    }
    
    // Accuracy : Iou1 > 0.5 == true positive ...
    for (int i = 0; i < std::size(IoU); i++){
        if(IoU[i] > 0.5){
            std::cout<<"true positive"<<std::endl;
        }else {
            std::cout<<"NO true positive"<<std::endl;
        }
    }
    


    return 0;

}

// HELPER FUCNTIONS
//std::vector<cv::Point>
 void parser(const std::string& path , std::vector<cv:: Point2f>& sugar, std::vector<cv:: Point2f>& mustard, std::vector<cv:: Point2f>& power_drill){

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

            sugar[0] = cv::Point2f(std::stoi(words[1]), std::stoi(words[2]));
            sugar[1] = cv::Point2f(std::stoi(words[3]), std::stoi(words[4]));

        }else if(words[0] == "006_mustard_bottle"){

            mustard[0] = cv::Point2f(std::stoi(words[1]), std::stoi(words[2]));
            mustard[1] = cv::Point2f(std::stoi(words[3]), std::stoi(words[4]));

        }else if (words[0] == "035_power_drill")
        {
            power_drill[0] = cv::Point2f(std::stoi(words[1]), std::stoi(words[2]));
            power_drill[1] = cv::Point2f(std::stoi(words[3]), std::stoi(words[4]));
        }
        
        // closed automatically the file open thanks to the distructor of ifstream
        
    }

}

 void printValue(std::vector<cv:: Point2f> v1, std::vector<cv:: Point2f> v2, std::vector<cv:: Point2f> v3){

    std::cout << "Sugar Coord:" << std::endl;
    for (const auto& p : v1) {
        std::cout << "(" << p.x << ", " << p.y << ") ";
    }
    std::cout << std::endl;

    std::cout << "Mustard Coord:" << std::endl;
    for (const auto& p : v2) {
        std::cout << "(" << p.x << ", " << p.y << ") ";
    }
    std::cout << std::endl;

    std::cout << "Power Drill Coord:" << std::endl;
    for (const auto& p : v3) {
        std::cout << "(" << p.x << ", " << p.y << ")";
    }
    std::cout << "\n\n";

}
double PerformanceMetrics::compute_mIoU();